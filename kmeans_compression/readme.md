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

![koala](./img/koala.jpg)

`K = 2`

![koala2](./img/koala2.jpg)

`K = 5`

![koala5](./img/koala5.jpg)

`K = 10`

![koala10](./img/koala10.jpg)

`K = 15`

![koala15](./img/koala15.jpg)

`K = 20`

![koala20](./img/koala20.jpg)
