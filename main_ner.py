from run_kbert_ner import main


class NERKBERTArgs:
    def __init__(self) -> None:
        self.pretrained_model_path = "./models/pytorch_model.bin"
        self.config_path = "./models/bert_config.json"
        self.vocab_path = "./models/vocab.txt"
        self.train_path = "./datasets/medical_ner/train.tsv"
        self.dev_path = "./datasets/medical_ner/dev.tsv"
        self.test_path = "./datasets/medical_ner/test.tsv"
        self.epochs_num = 5
        self.batch_size = 16
        self.kg_name = "Medical"  # "brain/kgs/medical_kg_en.spo"
        self.output_model_path = "./outputs/kbert_medical_ner_Medical.bin"


if __name__ == "__main__":
    args = NERKBERTArgs()
    main(args)
