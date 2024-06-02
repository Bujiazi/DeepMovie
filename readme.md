# DeepMovie

Official implementation of the AI3602 course project DeepMovie. It is a plug-and-play deep nueral network training pipeline turning most deep learning base modules into a movie recommender system.

Here is our project poster:

<iframe src="/poster.pdf" width="100%" height="600px">
</iframe>

### results(RMSE)

| model | movielens-1m | movielens-10m |
|-------|--------------|---------------|
| CNN           |   0.8733   |   0.7970   |
| CNN_KAN       |   0.8725   |   0.7941   |
| LSTM          |   0.8675   |   0.7959   |
| ResNet        |   0.8658   |   0.7931   |
| Transformer   | __0.8601__ | __0.7883__ |
