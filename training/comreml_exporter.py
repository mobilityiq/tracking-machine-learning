import coremltools as ct
import numpy as np
import json

class CoreMLExporter:

    @stationary
    def export_model_to_coreml(model, mean_timestamp, mean_x, mean_y, mean_z, mean_mx, mean_my, mean_mz, std_timestamp, std_x, std_y, std_z, std_mx, std_my, std_mz):
        mean = np.load('../model/mean.npy')
        std = np.load('../model/std.npy')

        mean_timestamp = mean[0]
        mean_x = mean[1]
        mean_y = mean[2]
        mean_z = mean[3]
        mean_mx = mean[4]
        mean_my = mean[5]
        mean_mz = mean[6]


        std_timestamp = std[0]
        std_x = std[1]
        std_y = std[2]
        std_z = std[3]
        std_mx = std[4]
        std_my = std[5]
        std_mz = std[6]

        # Create a dictionary to hold the metadata
        metadata = {
            'mean': [mean_timestamp, mean_x, mean_y, mean_z, mean_mx, mean_my, mean_mz],
            'std': [std_timestamp, std_x, std_y, std_z, std_mx, std_my, std_mz],
            'labels': ['driving','cycling','train','bus','metro', 'stationary', 'running', 'walking']
        }


        # Convert the model to Core ML format with a single input
        # input_shape = (1, features.shape[1])
        # input_feature = ct.TensorType(shape=input_shape)

        # coreml_model = ct.convert(model, inputs=[input_feature], source='tensorflow')
        coreml_model = ct.convert(model)

        # Convert the metadata dictionary to JSON string
        metadata_json = json.dumps(metadata)

        # Add the metadata to the model as user-defined metadata
        coreml_model.user_defined_metadata['preprocessing_metadata'] = metadata_json

        # Set the prediction_type to "probability"
        coreml_model.user_defined_metadata['prediction_type'] = 'probability'

        # Save the Core ML model
        coreml_model.save('../model/TransitModePredictor.mlmodel')