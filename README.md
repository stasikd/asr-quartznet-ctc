# asr-quartznet-ctc

- запуск обучения: `python run_quartznet_ctc.py`

![init model](https://github.com/stasikd/asr-quartznet-ctc/blob/main/assets/init_model.png)

- запуск обучения с предобученного чекпоинта: `python run_quartznet_ctc.py model.init_weights=/home/asr/data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt`

![from checkpoint](https://github.com/stasikd/asr-quartznet-ctc/blob/main/assets/from_checkpoint.png)

- кривые, показывающие переобучение модели со случайными весами на сэмпле длительностью ~10 минут

![loss](https://github.com/stasikd/asr-quartznet-ctc/blob/main/assets/loss.png)

![wer](https://github.com/stasikd/asr-quartznet-ctc/blob/main/assets/wer.png)

