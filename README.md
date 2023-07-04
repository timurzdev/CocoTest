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

### metric

- В качестве метрики качества была выбрана mAP
- Она учитывает и точность, и полноту
- Она учитывает взаимодействие между объектами разных классов

```python
    def on_validation_epoch_end(self):
    cocoDt = self.cocoGt.loadRes(self.results)
    cocoEval = COCOeval(self.cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # Get the Average Precision (AP) score
    avg_precision = cocoEval.stats[0]
    print(type(avg_precision))
    self.log('val_mAP', avg_precision)
```