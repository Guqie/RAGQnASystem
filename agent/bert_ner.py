"""可选的 BERT 对齐器。权重不在仓库里时返回 None，系统继续使用词典。"""

import pickle
from pathlib import Path


def load_bert_aligner(root: str):
    root_path = Path(root)
    model_dir = root_path / "model" / "chinese-roberta-wwm-ext"
    weights = root_path / "model" / "best_roberta_rnn_model_ent_aug.pt"
    tag_file = root_path / "tmp_data" / "tag2idx.npy"
    if not (model_dir.is_dir() and weights.exists() and tag_file.exists()):
        return None
    try:
        import torch
        from transformers import BertTokenizer

        import ner_model as zwk
    except Exception:
        return None
    if not all(hasattr(zwk, name) for name in ("rule_find", "tfidf_alignment", "Bert_Model", "get_ner_result")):
        return None

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    try:
        with open(tag_file, "rb") as handle:
            tag2idx = pickle.load(handle)
        idx2tag = list(tag2idx)
        tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        model = zwk.Bert_Model(str(model_dir), hidden_size=128, tag_num=len(tag2idx), bi=True)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device)
        model.eval()
        rule = zwk.rule_find()
        aligner = zwk.tfidf_alignment()
    except Exception:
        return None

    def align(text: str):
        with torch.no_grad():
            return zwk.get_ner_result(model, tokenizer, text, rule, aligner, device, idx2tag) or {}

    return align


def recognizer_backend(recognizer) -> str:
    sample = getattr(recognizer, "dictionary", recognizer)
    count = len(getattr(sample, "automata", {}) or {})
    if recognizer.__class__.__name__ == "HybridRecognizer":
        return f"BERT + 词典（{count} 类）"
    return f"词典匹配（{count} 类）"
