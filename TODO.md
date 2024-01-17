# General TODOs
- Mettere il loading del codice da github invece che drive
- Trees detection : 2011 
- Finire presentazione : 2011 e possibili miglioramenti futuri
- Fare test esaustivi
- Test con folder contenenti piu immagini

# Green areas detection
- Cambiare il nome di `utils.utils` in `utils.inference` (piccolo refactoring)
- `crop_size` e `step`?
- `batch_size` non funziona come previsto
- Migliorare efficenza della funzione `utils.predict_on_img`, cambiando come vengono calcolati i crop (prendere spunto da `dataset_handler`)
- In `utils.predict_on_img`, implementare il gaussian smoothing in modo che faccia media pesata invece che media normale.
- Provare Segment anything model
- Migliorare tqdm di `utils.predict_on_img`


# Trees detection
- ALREADY CHECKED. If there is an error during the process, the temporary files are not deleted.
- Settare conf e iou thres per il train.py di yolov7, in modo da avere metriche piu consistenti durante il training. Guardare `test.py`: di default `conf=0.0001` e `iou_thres=0.6`.
- `CROP_SIZE` e `STEP` metterle come variabili globali comuni a tutti i file (invece di ridefinirle indipendenti in ogni file)