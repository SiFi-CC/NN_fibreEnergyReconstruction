
# The following accepts any number of dimensions, which is controlled by kernel_size

class Conv(Layer):
    def __init__(self, filters, kernel_size, padding='VALID', **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size #must be a tuple!!!!
        self.padding=padding

        super(Conv, self).__init__(**kwargs)

    #using channels last!!!
    def build(self, input_shape):
        spatialDims = len(self.kernel_size)
        allDims = len(input_shape)
        assert allDims == spatialDims + 2 #spatial dimensions + batch size + channels

        kernelShape = self.kernel_size + (input_shape[-1], self.filters)
            #(spatial1, spatial2,...., spatialN, input_channels, output_channels)

        biasShape = tuple(1 for _ in range(allDims-1)) + (self.filters,)


        self.kernel = self.add_weight(name='kernel', 
                                      shape=kernelShape
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                    shape = biasShape, 
                                    initializer='zeros',
                                    trainable=True)
        self.built = True

    def call(self, inputs):
        results = tf.nn.convolution(inputs, self.kernel, padding=self.padding)
        return results + self.bias

    def compute_output_shape(self, input_shape)
        sizes = input_shape[1:-1]

        if self.padding='VALID' or self.padding='valid':
            sizes = [s - kSize + 1 for s, kSize in zip(sizes, self.kernel_size)]

        return input_shape[:1] + sizes + (self.filters,)
