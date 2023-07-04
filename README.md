# CocoTest

### run training

```bash
python main.py --train --logger_name logger_name --data_folder ./data/coco2017 --batch_size 16 
```

### build docker image

```bash
docker build -t detection_server .   
```

### run docker container

```bash
docker run --rm -it -p 7777:7777 detection_server                                                  
```

### run tests

```bash
python tests.py --file <path_to_image> --url http://0.0.0.0:7777/predict
```