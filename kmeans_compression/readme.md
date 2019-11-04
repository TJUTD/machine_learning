#  K-means clustering on images

```html
Version: Python 3.6.0 
```

Command to run on terminal:

```html
python kmeans_compression.py --inputpath <image file path> --k <number of cluster> --outputpath <output directory> --method <'show'(default), 'save', 'ratio'>
```

If `method == 'show'`,  plot the compressed image.

If `method == 'save'`,  save the compressed image silently.

If `method == 'ratio'`,  calculate the compression ratio over 50 trials.


Result:

Original picture

![koala](./img/koala.jpg){:height="50%" width="50%"}

`K = 2`

![koala_k_2](./img/koala_k_2.JPG){:height="50%" width="50%"}

`K = 5`

![koala_k_5](./img/koala_k_5.JPG){:height="50%" width="50%"}

`K = 10`

![koala_k_10](./img/koala_k_10.JPG){:height="50%" width="50%"}

`K = 15`

![koala_k_15](./img/koala_k_15.JPG){:height="50%" width="50%"}

`K = 20`

![koala_k_20](./img/koala_k_20.JPG){:height="50%" width="50%"}
