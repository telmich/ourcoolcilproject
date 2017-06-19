Did not work, but I think I know why: I only retrained the last layer, so the features were not re-trained. And because our astronomy images have different interesting features than general stuff like cats and dogs that the inception features were trained on. So the approach with retraining only the last layer cannot work. Therefore we probably need a smaller CNN that we can fully train ourselves.

For the record, here's what I did here:

I followed https://www.tensorflow.org/tutorials/image_retraining.

Note that some stuff there is out of date. Below are the commands that actually worked for me (as opposed to the commands documented there).

Details:

Prepare images
--------------

```
./retrain_prepare_images.py ./retrain_images
for d in ./retrain_imgs/*; do mogrify -format jpg "$d/*.png"; done  # their retrain code eats only jpg
```

Setup
-----

set up bazel according to https://bazel.build/versions/master/docs/install-ubuntu.html#install-on-ubuntu + **had to install `openjdk-8-jdk`**
```
pip install tensorflow
git submodule add https://github.com/tensorflow/tensorflow # + check out the most recent stable version
# note: you probably want something other than add :-)
cd tensorflow
./configure
```

Build the stuff
---------------

```
bazel build --config opt tensorflow/examples/image_retraining:retrain tensorflow/examples/label_image:label_image
# go for a long walk, maybe to a shop where they sell RAM
```

Run the retrain
---------------

```
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ./retrain_images
# go for a long walk again
```

Test it
-------

```
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
                                          --output_layer=final_result \
                                          --input_layer=Mul \
                                          --image=/ABSOLUTE/path/to/image.jpg
```
