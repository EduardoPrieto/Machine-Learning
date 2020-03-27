import tensorflowjs
import tensorflow as tf

tensorflowjs_converter \
    --input_format=checkpoint \
    --output_node_names='Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model