#!/usr/bin/env python
"""
Custom ONNX export for YOLOv8 compatible with DepthAI YoloDetectionNetwork.

Output format per head: [1, 4 + num_classes, H, W]
  - Channels 0:4  = DFL-decoded bbox (LTRB distances from grid cell)
  - Channels 4:   = raw class logits (NO sigmoid — DepthAI applies it)

DFL decode (softmax + weighted projection) stays in the ONNX graph.
Sigmoid and ReduceMax are removed — DepthAI handles those internally.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class RawDetectWrapper(nn.Module):
    """Wraps ultralytics Detect head for DepthAI YoloDetectionNetwork.

    Keeps DFL bbox decoding in the graph (4 output channels).
    Outputs raw class logits without sigmoid (DepthAI applies sigmoid).
    No ReduceMax confidence channel.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Get detect head (last module in the sequential model)
        self.detect = model.model[-1]
        self.nc = self.detect.nc
        self.nl = self.detect.nl
        self.reg_max = self.detect.reg_max
        self.stride = self.detect.stride

        # DFL layer for bbox decoding
        self.dfl = self.detect.dfl

        # Box and class prediction heads
        self.cv2 = self.detect.cv2
        self.cv3 = self.detect.cv3

    def forward(self, x):
        features = self._run_backbone(x)

        head_outputs = []
        for i in range(self.nl):
            feat = features[i]

            # Box regression: raw logits -> DFL decoded LTRB
            box_raw = self.cv2[i](feat)  # [b, 4*reg_max, h, w]
            b, _, h, w = box_raw.shape

            if self.reg_max > 1:
                # DFL decode: softmax over reg_max bins, weighted sum
                box_dfl = box_raw.view(b, 4, self.reg_max, h * w).permute(0, 2, 1, 3)
                box_dfl = F.softmax(box_dfl, dim=1)
                box_dfl = self.dfl.conv(box_dfl)[:, 0]  # [b, 4, h*w]
            else:
                box_dfl = box_raw.view(b, 4, h * w)

            # Class logits -> sigmoid -> class probs
            cls_raw = self.cv3[i](feat)  # [b, nc, h, w]
            cls_raw = cls_raw.view(b, self.nc, h * w)  # [b, nc, h*w]
            cls_prob = torch.sigmoid(cls_raw)  # [b, nc, h*w]

            # Max confidence across classes (objectness surrogate)
            max_conf = cls_prob.max(dim=1, keepdim=True).values  # [b, 1, h*w]

            # Output: [bbox_dfl(4), max_conf(1), sigmoid_cls(nc)]
            combined = torch.cat([box_dfl, max_conf, cls_prob], dim=1)  # [b, 5+nc, h*w]
            head_outputs.append(combined.view(b, 5 + self.nc, h, w))

        return tuple(head_outputs)

    def _run_backbone(self, x):
        """Run all layers except the detect head, return intermediate features."""
        model_seq = self.model.model
        save_set = self.model.save  # layer indices whose outputs are needed later
        detect_idx = len(model_seq) - 1
        y = []

        for i, m in enumerate(model_seq):
            if i == detect_idx:
                break
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in save_set else None)

        # Collect feature maps for detect head
        detect = model_seq[detect_idx]
        if isinstance(detect.f, int):
            features = [y[detect.f]]
        else:
            features = [x if j == -1 else y[j] for j in detect.f]

        return features


def export_onnx_raw(model_path, imgsz, output_path):
    """Export YOLOv8 .pt model to ONNX with raw logits for DepthAI.

    Args:
        model_path: Path to ultralytics .pt model.
        imgsz: [width, height] input size.
        output_path: Path to save the ONNX file.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO

    width, height = imgsz

    # Load model
    yolo = YOLO(model_path)
    model = yolo.model.eval()

    # Wrap with raw output
    wrapper = RawDetectWrapper(model)
    wrapper.eval()

    # Create dummy input
    dummy = torch.zeros(1, 3, height, width)

    # Separate output per detection head (preserves grid dimensions)
    output_names = [f"output{i+1}_yolov6r2" for i in range(wrapper.nl)]

    # Test forward pass
    with torch.no_grad():
        test_outputs = wrapper(dummy)
        for i, out in enumerate(test_outputs):
            print(f"  {output_names[i]}: shape={list(out.shape)}")

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=12,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
    )
    print(f"  ONNX exported: {output_path}")

    # Verify no sigmoid in output subgraph
    _verify_raw_outputs(output_path)

    return output_path


def _verify_raw_outputs(onnx_path):
    """Verify output structure. Sigmoid is expected (class probs), Softmax is not."""
    import onnx

    model = onnx.load(onnx_path)

    # Build map: tensor name -> producing node
    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    ok = True
    for out in model.graph.output:
        node = output_to_node.get(out.name)
        if not node:
            print(f"  WARNING: Output {out.name} has no producing node")
            ok = False
            continue

        # Check the output node and its immediate inputs for forbidden ops
        forbidden = {"Softmax"}
        if node.op_type in forbidden:
            print(f"  FAIL: Output {out.name} directly from {node.op_type}!")
            ok = False
            continue

        # Check inputs of the output node (one level deep)
        has_forbidden = False
        for inp_name in node.input:
            inp_node = output_to_node.get(inp_name)
            if inp_node and inp_node.op_type in forbidden:
                print(f"  FAIL: Output {out.name} <- {node.op_type} <- {inp_node.op_type}")
                has_forbidden = True
                ok = False

        if not has_forbidden:
            # Get output shape from value_info or output
            shape_str = ""
            for o in model.graph.output:
                if o.name == out.name and o.type.tensor_type.shape.dim:
                    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
                    shape_str = f" shape={dims}"
            print(f"  OK: {out.name} <- {node.op_type}{shape_str}")

    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 to raw-logit ONNX")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 352],
                        help="Input size as width height")
    parser.add_argument("--output", default="output/best.onnx",
                        help="Output ONNX path")
    args = parser.parse_args()

    export_onnx_raw(args.model, args.imgsz, args.output)
