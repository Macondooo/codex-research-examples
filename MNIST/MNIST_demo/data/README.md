# MNIST CSV files

MNIST contains grayscale images of handwritten digits from `0` to `9`. These files use Joseph Redmon's CSV conversion of the original dataset.

| File | Rows | Use |
| --- | ---: | --- |
| `raw/mnist_train.csv` | 60,000 | model fitting |
| `raw/mnist_test.csv` | 10,000 | final evaluation |

The files have no header row. Each row contains 785 integers: the digit label first, followed by 784 pixels in row-major order for a `28 × 28` image. Pixel values range from `0` to `255`.

Source: [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/). Direct files: [training set](https://data.pjreddie.com/files/mnist_train.csv) and [test set](https://data.pjreddie.com/files/mnist_test.csv).
