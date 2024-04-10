import torch
from torch import nn

    def model_fn_v1(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        ######
        tf.logging.info('*********************************** MbertPcnnModel V1 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_representation = molecule_bert.get_pooled_output()

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_representation, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')
        #####

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.mean_squared_error(y, predictions)

            # self.y = y
            # self.predictions = predictions

            tvars = tf.trainable_variables()

            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                if self.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            def metric_fn(loss, y_true, y_pred):
                mean_loss = tf.metrics.mean(values=loss)

                g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
                g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

                f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
                f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

                g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
                f, update_op2 = tf.metrics.mean(f)

                cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

                return {
                    "mse": mean_loss,
                    "cindex": cindex,
                }

            eval_metrics = (metric_fn, [loss, y, predictions])

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = create_optimizer(
                    loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, self.use_tpu)

            else:
                train_op = None

        else:
            loss = None
            train_op = None
            eval_metrics = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "predictions": predictions,
                "gold": y,
                "xd": xd,
                "xt": xt,
            },
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                {"predictions": predictions})})

        return output_spec

class MbertPcnnModel(nn.Module):
    def forward(self, xd, xd_mask, xt):

        bert_output = self.bert(input_ids = xd, attetion_mask=xd_mask)
        molecule_representation = bert_output.pooler_output

        concat_z = torch.cat([molecule_representation, cnn_output], 1)
        z = torch.relu(self.fc1(concat_z))