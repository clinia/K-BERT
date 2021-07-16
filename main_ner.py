from run_kbert_ner import main


class NERKBERTArgs:
    def __init__(self) -> None:

        """
        Choose model:

        CHINESE MODEL : 0

        ENGLISH MODEL : 1

        """
        model = 1

        if model == 0:
            self.pretrained_model_path = "./models/pytorch_model.bin"
            self.config_path = "./models/bert_config.json"
            self.vocab_path = "./models/vocab.txt"
            self.train_path = "./datasets/medical_ner/train.tsv"
            self.dev_path = "./datasets/medical_ner/dev.tsv"
            self.test_path = "./datasets/medical_ner/test.tsv"
            self.epochs_num = 5
            self.batch_size = 16
            self.kg_name = "Medical"
            self.output_model_path = "./outputs/kbert_medical_ner_Medical.bin"
            self.labels_path = "datasets/clinia_ner/ser_bus/labels.csv"

        if model == 1:
            self.pretrained_model_path = "./models/en/bert_base_uncased/pytorch_model.bin"
            self.config_path = "./models/en/bert_base_uncased/config.json"
            self.vocab_path = "./models/en/bert_base_uncased/vocab.txt"
            self.tokenizer_config = "./models/en/bert_base_uncased/tokeniser_config.json"
            self.tokenizer = ".model/en/bert_base_uncased/tokenizer.json"
            self.train_path = "./datasets/clinia_ner/ser_bus/validation/data.csv"  ###*** to change to train
            self.dev_path = "./datasets/clinia_ner/ser_bus/validation/data.csv"
            self.test_path = "./datasets/clinia_ner/ser_bus/test/data.cvs"
            self.epochs_num = 5
            self.batch_size = 16
            self.kg_name = "brain/kgs/medical_kg_en.spo"
            self.output_model_path = "./outputs/kbert_medical_ner_Medical_en.bin"
            self.labels_path = "datasets/clinia_ner/ser_bus/labels.csv"


if __name__ == "__main__":
    args = NERKBERTArgs()
    main(args)
