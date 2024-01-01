from argparse import ArgumentParser

import torch
from tensorflow.python.tools.inspect_checkpoint import py_checkpoint_reader

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--tf_checkpoint", required=True, type=str, help="Path to original Tensorflow checkpoint.")
parser.add_argument(
    "--pt_checkpoint_path", default="raft_smurf.pt", type=str, help="Path of converted PyTorch checkpoint."
)

# Get arguments
args = parser.parse_args()

# TF keys to PyTorch
TFtoPT = {
    # Context encoder
    "feature_model/cnet/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.convnormrelu.0.weight",
    "feature_model/cnet/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.convnormrelu.0.bias",
    "feature_model/cnet/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.conv.weight",
    "feature_model/cnet/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.conv.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.0.convnormrelu1.0.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.0.convnormrelu1.0.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.0.convnormrelu2.0.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.0.convnormrelu2.0.bias",
    "feature_model/cnet/layer1/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.1.convnormrelu1.0.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.1.convnormrelu1.0.bias",
    "feature_model/cnet/layer1/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.1.convnormrelu2.0.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer1.1.convnormrelu2.0.bias",
    "feature_model/cnet/layer2/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.convnormrelu1.0.weight",
    "feature_model/cnet/layer2/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.convnormrelu1.0.bias",
    "feature_model/cnet/layer2/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.convnormrelu2.0.weight",
    "feature_model/cnet/layer2/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.convnormrelu2.0.bias",
    "feature_model/cnet/layer2/layer_with_weights-0/downsample/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.downsample.0.weight",
    "feature_model/cnet/layer2/layer_with_weights-0/downsample/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.0.downsample.0.bias",
    "feature_model/cnet/layer2/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.1.convnormrelu1.0.weight",
    "feature_model/cnet/layer2/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.1.convnormrelu1.0.bias",
    "feature_model/cnet/layer2/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.1.convnormrelu2.0.weight",
    "feature_model/cnet/layer2/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer2.1.convnormrelu2.0.bias",
    "feature_model/cnet/layer3/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.convnormrelu1.0.weight",
    "feature_model/cnet/layer3/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.convnormrelu1.0.bias",
    "feature_model/cnet/layer3/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.convnormrelu2.0.weight",
    "feature_model/cnet/layer3/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.convnormrelu2.0.bias",
    "feature_model/cnet/layer3/layer_with_weights-0/downsample/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.downsample.0.weight",
    "feature_model/cnet/layer3/layer_with_weights-0/downsample/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.0.downsample.0.bias",
    "feature_model/cnet/layer3/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.1.convnormrelu1.0.weight",
    "feature_model/cnet/layer3/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.1.convnormrelu1.0.bias",
    "feature_model/cnet/layer3/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.1.convnormrelu2.0.weight",
    "feature_model/cnet/layer3/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "context_encoder.layer3.1.convnormrelu2.0.bias",
    # Feature encoder
    "feature_model/fnet/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.convnormrelu.0.weight",
    "feature_model/fnet/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.convnormrelu.0.bias",
    "feature_model/fnet/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.conv.weight",
    "feature_model/fnet/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.conv.bias",
    "feature_model/fnet/layer1/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.0.convnormrelu1.0.weight",
    "feature_model/fnet/layer1/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.0.convnormrelu1.0.bias",
    "feature_model/fnet/layer1/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.0.convnormrelu2.0.weight",
    "feature_model/fnet/layer1/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.0.convnormrelu2.0.bias",
    "feature_model/fnet/layer1/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.1.convnormrelu1.0.weight",
    "feature_model/fnet/layer1/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.1.convnormrelu1.0.bias",
    "feature_model/fnet/layer1/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.1.convnormrelu2.0.weight",
    "feature_model/fnet/layer1/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer1.1.convnormrelu2.0.bias",
    "feature_model/fnet/layer2/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.convnormrelu1.0.weight",
    "feature_model/fnet/layer2/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.convnormrelu1.0.bias",
    "feature_model/fnet/layer2/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.convnormrelu2.0.weight",
    "feature_model/fnet/layer2/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.convnormrelu2.0.bias",
    "feature_model/fnet/layer2/layer_with_weights-0/downsample/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.downsample.0.weight",
    "feature_model/fnet/layer2/layer_with_weights-0/downsample/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.0.downsample.0.bias",
    "feature_model/fnet/layer2/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.1.convnormrelu1.0.weight",
    "feature_model/fnet/layer2/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.1.convnormrelu1.0.bias",
    "feature_model/fnet/layer2/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.1.convnormrelu2.0.weight",
    "feature_model/fnet/layer2/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer2.1.convnormrelu2.0.bias",
    "feature_model/fnet/layer3/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.convnormrelu1.0.weight",
    "feature_model/fnet/layer3/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.convnormrelu1.0.bias",
    "feature_model/fnet/layer3/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.convnormrelu2.0.weight",
    "feature_model/fnet/layer3/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.convnormrelu2.0.bias",
    "feature_model/fnet/layer3/layer_with_weights-0/downsample/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.downsample.0.weight",
    "feature_model/fnet/layer3/layer_with_weights-0/downsample/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.0.downsample.0.bias",
    "feature_model/fnet/layer3/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.1.convnormrelu1.0.weight",
    "feature_model/fnet/layer3/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.1.convnormrelu1.0.bias",
    "feature_model/fnet/layer3/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.1.convnormrelu2.0.weight",
    "feature_model/fnet/layer3/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "feature_encoder.layer3.1.convnormrelu2.0.bias",
    # Update block
    "flow_model/update_block/encoder/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.conv.0.weight",
    "flow_model/update_block/encoder/conv/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.conv.0.bias",
    "flow_model/update_block/encoder/convc1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convcorr1.0.weight",
    "flow_model/update_block/encoder/convc1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convcorr1.0.bias",
    "flow_model/update_block/encoder/convc2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convcorr2.0.weight",
    "flow_model/update_block/encoder/convc2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convcorr2.0.bias",
    "flow_model/update_block/encoder/convf1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convflow1.0.weight",
    "flow_model/update_block/encoder/convf1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convflow1.0.bias",
    "flow_model/update_block/encoder/convf2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convflow2.0.weight",
    "flow_model/update_block/encoder/convf2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.motion_encoder.convflow2.0.bias",
    # Recurrent block
    "flow_model/update_block/gru/convq1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convq.weight",
    "flow_model/update_block/gru/convq1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convq.bias",
    "flow_model/update_block/gru/convq2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convq.weight",
    "flow_model/update_block/gru/convq2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convq.bias",
    "flow_model/update_block/gru/convr1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convr.weight",
    "flow_model/update_block/gru/convr1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convr.bias",
    "flow_model/update_block/gru/convr2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convr.weight",
    "flow_model/update_block/gru/convr2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convr.bias",
    "flow_model/update_block/gru/convz1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convz.weight",
    "flow_model/update_block/gru/convz1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru1.convz.bias",
    "flow_model/update_block/gru/convz2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convz.weight",
    "flow_model/update_block/gru/convz2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.recurrent_block.convgru2.convz.bias",
    # Head
    "flow_model/update_block/flow_head/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.flow_head.conv1.weight",
    "flow_model/update_block/flow_head/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.flow_head.conv1.bias",
    "flow_model/update_block/flow_head/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "update_block.flow_head.conv2.weight",
    "flow_model/update_block/flow_head/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "update_block.flow_head.conv2.bias",
    # Mask predictor
    "flow_model/update_block/mask/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "mask_predictor.convrelu.0.weight",
    "flow_model/update_block/mask/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "mask_predictor.convrelu.0.bias",
    "flow_model/update_block/mask/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "mask_predictor.conv.weight",
    "flow_model/update_block/mask/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE": "mask_predictor.conv.bias",
}


def main() -> None:
    # Make checkpoint reader
    reader = py_checkpoint_reader.NewCheckpointReader(args.tf_checkpoint)
    # Get dict of weights names and shapes
    var_to_shape = reader.get_variable_to_shape_map()
    # Init PyTorch state dict
    state_dict = {}
    # Convert weights
    for key, value in sorted(var_to_shape.items()):
        if key in TFtoPT.keys():
            if "kernel" in key:
                state_dict[TFtoPT[key]] = torch.from_numpy(reader.get_tensor(key)).permute(3, 2, 0, 1)
            else:
                state_dict[TFtoPT[key]] = torch.from_numpy(reader.get_tensor(key))
    # Save checkpoint
    torch.save(state_dict, args.pt_checkpoint_path)


if __name__ == "__main__":
    main()
