# Data Directory

Datasets are not committed to the repository.

Expected layout:

```text
data/
  GUE/
    prom/
      prom_core_tata/{train.csv,dev.csv,test.csv}
      prom_core_notata/{train.csv,dev.csv,test.csv}
      prom_core_all/{train.csv,dev.csv,test.csv}
      prom_300_tata/{train.csv,dev.csv,test.csv}
      prom_300_notata/{train.csv,dev.csv,test.csv}
      prom_300_all/{train.csv,dev.csv,test.csv}
    virus/
      covid/{train.csv,dev.csv,test.csv}
  TEB/
    train_processed_pseudogenes.csv
    valid_processed_pseudogenes.csv
    test_processed_pseudogenes.csv
    train_unprocessed_pseudogenes.csv
    valid_unprocessed_pseudogenes.csv
    test_unprocessed_pseudogenes.csv
```

CIFAR-10 and CIFAR-100 are downloaded automatically by `torchvision` into this
directory when running the vision scripts.

