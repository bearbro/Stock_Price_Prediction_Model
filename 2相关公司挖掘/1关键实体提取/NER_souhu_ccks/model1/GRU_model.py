import warnings
import tensorflow as tf
import keras
from keras.layers import Layer
from keras.layers import Input, Dropout, Dense, GRUCell, GRU, RNN
from keras import backend as K
from keras import initializers, activations, regularizers, constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.legacy import interfaces


class TCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(TCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3),
                                                name='recurrent_kernel_t',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)
        if self.use_bias:
            if not self.reset_after:
                self.recurrent_bias = None
            else:
                bias_shape = (3 * self.units,)
                self.bias = self.add_weight(shape=bias_shape,
                                            name='bias',
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
                self.recurrent_bias = K.flatten(self.bias)
        else:
            self.bias = None

        # update gate
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        # new gate
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:self.units * 3]

        if self.use_bias:
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:self.units * 3]
        else:
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None

        self.built = True
        # super(TCell, self).build(input_shape)

    def call(self, states, training=None):
        h_tm1 = states  # previous memory

        if (0 < self.recurrent_dropout < 1 and self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]

            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)

            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
                recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

            z = self.recurrent_activation(recurrent_z)
            r = self.recurrent_activation(recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:  # TGRU 选这个
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
                recurrent_h = r * recurrent_h
            else:  # default
                recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(recurrent_h)
        else:

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(recurrent_z)
            r = self.recurrent_activation(recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:3 * self.units]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:3 * self.units])

            hh = self.activation(recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh  # todo why
        # h = (1-z) * h_tm1 + z * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class LGRUCell(Layer):
    """Cell class for the MyGRU layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=True,  # False,
                 tcell_num=0,
                 **kwargs):
        super(LGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        # tcell
        self.t_cell_num = tcell_num
        self.t_recurrent_activation = activations.get(recurrent_activation)
        self.t_recurrent_initializer = initializers.get(recurrent_initializer)
        self.t_recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.t_recurrent_constraint = constraints.get(recurrent_constraint)
        self.t_recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.t__recurrent_dropout_mask = [None] * tcell_num

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 5),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.t_recurrent_kernel = []
        for i in range(self.t_cell_num):
            self.t_recurrent_kernel.append(self.add_weight(
                shape=(self.units, self.units * 3),
                name='t_recurrent_kernel_%i' % i,
                initializer=self.t_recurrent_initializer,
                regularizer=self.t_recurrent_regularizer,
                constraint=self.t_recurrent_constraint))

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (5 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 5 * self.units)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
                self.t_bias = []
                self.t_recurrent_bias = []
                for i in range(self.t_cell_num):
                    self.t_bias.append(self.add_weight(shape=(3 * self.units,),
                                                       name='t_bias_%d' % i,
                                                       initializer=self.bias_initializer,
                                                       regularizer=self.bias_regularizer,
                                                       constraint=self.bias_constraint))
                    self.t_recurrent_bias.append(K.flatten(self.t_bias[i]))

        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.t_recurrent_kernel_z = []
        self.t_recurrent_kernel_r = []
        self.t_recurrent_kernel_h = []
        for i in range(self.t_cell_num):
            self.t_recurrent_kernel_z.append(self.t_recurrent_kernel[i][:, :self.units])
            self.t_recurrent_kernel_r.append(
                self.t_recurrent_kernel[i][:, self.units:self.units * 2])
            self.t_recurrent_kernel_h.append(self.t_recurrent_kernel[i][:,
                                             self.units * 2:self.units * 3])
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:self.units * 3]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:self.units * 3]

        # myl
        self.kernel_myl = self.kernel[:, self.units * 3:self.units * 4]
        self.recurrent_kernel_myl = self.recurrent_kernel[:, self.units * 3:self.units * 4]
        # myl2
        self.kernel_myl2 = self.kernel[:, self.units * 4:]
        # self.recurrent_kernel_myl2 = self.recurrent_kernel[:, self.units * 4:]
        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:self.units * 3]
            self.input_bias_myl = self.input_bias[self.units * 3:self.units * 4]
            self.input_bias_myl2 = self.input_bias[self.units * 4:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:self.units * 3]
                self.recurrent_bias_myl = self.recurrent_bias[self.units * 3:self.units * 4]
                # self.recurrent_bias_myl2 = self.recurrent_bias[self.units * 4:]
                self.t_recurrent_bias_z = []
                self.t_recurrent_bias_r = []
                self.t_recurrent_bias_h = []
                for i in range(self.t_cell_num):
                    self.t_recurrent_bias_z.append(self.t_recurrent_bias[i][:self.units])
                    self.t_recurrent_bias_r.append((
                        self.t_recurrent_bias[i][self.units: self.units * 2]))
                    self.t_recurrent_bias_h.append(
                        self.t_recurrent_bias[i][self.units * 2:self.units * 3])
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            self.input_bias_myl = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
                self.recurrent_bias_myl = None

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=5)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
                inputs_myl = inputs * dp_mask[3]
                inputs_myl2 = inputs * dp_mask[4]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
                inputs_myl = inputs
                inputs_myl2 = inputs

            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            x_myl = K.dot(inputs_myl, self.kernel_myl)
            x_myl2 = K.dot(inputs_myl2, self.kernel_myl2)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.input_bias_z)
                x_r = K.bias_add(x_r, self.input_bias_r)
                x_h = K.bias_add(x_h, self.input_bias_h)
                x_myl = K.bias_add(x_myl, self.input_bias_myl)
                x_myl2 = K.bias_add(x_myl2, self.input_bias_myl2)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
                h_tm1_myl = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_myl = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
            recurrent_myl = K.dot(h_tm1_myl, self.recurrent_kernel_myl)
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
                recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)
                recurrent_myl = K.bias_add(recurrent_myl, self.recurrent_bias_myl)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            myl = self.recurrent_activation(x_myl + recurrent_myl)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:  # MyGRU 选这个
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
                recurrent_h = r * recurrent_h
            else:  # default
                recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h) + myl * x_myl2
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, self.input_bias)
            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:3 * self.units]
            x_myl = matrix_x[:, 3 * self.units:4 * self.units]
            x_myl2 = matrix_x[:, 4 * self.units:5 * self.units]
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]
            if self.reset_after:
                recurrent_myl = matrix_inner[:, self.units * 3: 4 * self.units]
            else:
                recurrent_myl = K.dot(h_tm1, self.recurrent_kernel[:, 3 * self.units:4 * self.units])

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            myl = self.recurrent_activation(x_myl + recurrent_myl)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:3 * self.units]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:3 * self.units])

            hh = self.activation(x_h + recurrent_h) + myl * x_myl2

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh  # todo why
        # h = (1-z) * h_tm1 + z * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        # tcell
        for i in range(self.t_cell_num):
            h_tm1 = h
            if (0 < self.t_recurrent_dropout < 1 and
                    self.t__recurrent_dropout_mask[i] is None):
                self.t__recurrent_dropout_mask[i] = _generate_dropout_mask(
                    K.ones_like(h_tm1),
                    self.t_recurrent_dropout,
                    training=training,
                    count=3)
            rec_dp_mask = self.t__recurrent_dropout_mask[i]
            if self.implementation == 1:
                if 0. < self.recurrent_dropout < 1.:
                    h_tm1_z = h_tm1 * rec_dp_mask[0]
                    h_tm1_r = h_tm1 * rec_dp_mask[1]
                    h_tm1_h = h_tm1 * rec_dp_mask[2]

                else:
                    h_tm1_z = h_tm1
                    h_tm1_r = h_tm1
                    h_tm1_h = h_tm1

                recurrent_z = K.dot(h_tm1_z, self.t_recurrent_kernel_z[i])
                recurrent_r = K.dot(h_tm1_r, self.t_recurrent_kernel_r[i])

                if self.reset_after and self.use_bias:
                    recurrent_z = K.bias_add(recurrent_z, self.t_recurrent_bias_z[i])
                    recurrent_r = K.bias_add(recurrent_r, self.t_recurrent_bias_r[i])

                z = self.recurrent_activation(recurrent_z)
                r = self.recurrent_activation(recurrent_r)

                # reset gate applied after/before matrix multiplication
                if self.reset_after:  # TGRU 选这个
                    recurrent_h = K.dot(h_tm1_h, self.t_recurrent_kernel_h[i])
                    if self.use_bias:
                        recurrent_h = K.bias_add(recurrent_h, self.t_recurrent_bias_h[i])
                    recurrent_h = r * recurrent_h
                else:  # default
                    recurrent_h = K.dot(r * h_tm1_h, self.t_recurrent_bias_h[i])

                hh = self.activation(recurrent_h)
            else:

                if 0. < self.recurrent_dropout < 1.:
                    h_tm1 *= rec_dp_mask[0]

                if self.reset_after:
                    # hidden state projected by all gate matrices at once
                    matrix_inner = K.dot(h_tm1, self.t_recurrent_kernel[i])
                    if self.use_bias:
                        matrix_inner = K.bias_add(matrix_inner, self.t_recurrent_bias[i])
                else:
                    # hidden state projected separately for update/reset and new
                    matrix_inner = K.dot(h_tm1,
                                         self.t_recurrent_kernel[i][:, :2 * self.units])

                recurrent_z = matrix_inner[:, :self.units]
                recurrent_r = matrix_inner[:, self.units: 2 * self.units]

                z = self.recurrent_activation(recurrent_z)
                r = self.recurrent_activation(recurrent_r)

                if self.reset_after:
                    recurrent_h = r * matrix_inner[:, 2 * self.units:3 * self.units]
                else:
                    recurrent_h = K.dot(r * h_tm1,
                                        self.recurrent_kernel[i][:, 2 * self.units:3 * self.units])

                hh = self.activation(recurrent_h)

                # previous and candidate state mixed by update gate
            h = z * h_tm1 + (1 - z) * hh  # todo why
            # h = (1-z) * h_tm1 + z * hh

            if 0 < self.dropout + self.recurrent_dropout:
                if training is None:
                    h._uses_learning_phase = True

        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(LGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyGRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    There are two variants. The default one is based on 1406.1078v3 and
    has reset gate applied to hidden state before matrix multiplication. The
    other one is based on original 1406.1078v1 and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. Use `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).

    # References
        - [Learning Phrase Representations using RNN Encoder-Decoder for
           Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
        - [On the Properties of Neural Machine Translation:
           Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on
           Sequence Modeling](https://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=True,  # False,
                 tcell_num=0,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        cell = LGRUCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation,
                        reset_after=reset_after,
                        tcell_num=tcell_num
                        )
        super(MyGRU, self).__init__(cell,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    unroll=unroll,
                                    **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(MyGRU, self).call(inputs,
                                       mask=mask,
                                       training=training,
                                       initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(MyGRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)

# def modify_tt():
#     maxlen=256
#     learning_rate=5e-5
#     x1_in = Input(shape=(maxlen,768))  # 待识别句子输入
#
#     h=x1_in
#     outputs=MyGRU(output_dim=64, return_sequences=True,reset_after=True,tcell_num=10)(h)
#     outputs = Dense(units=2)(outputs)
#
#     model = keras.models.Model(x1_in, outputs, name='basic_bilstm_crf_model')
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate),loss='categorical_crossentropy',metrics=['acc'])
#     model.summary()
#
#     return model
#
#
#
#
# model=modify_tt()
# import numpy as np
#
# x=np.random.random((100,256,768))
# y=np.ones((100,256,2))
# model.fit(x,y,epochs=5,verbose=1)
#
#
# mmm=MyGRU(units=300,reset_after=True,tcell_num=5)
# y=mmm(tf.ones(shape=(100,256,768)))
# print(y)
# print(len(mmm.trainable_weights))
# print(mmm.trainable_weights)
