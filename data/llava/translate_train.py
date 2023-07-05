import json
import os
import numpy as np
import torch
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, \
    AutoModelForSeq2SeqLM
from tqdm import tqdm

mc4_lang_dist = {'af': 0.01974382519794401, 'am': 0.003251616352426345, 'ar': 1.0310711629267721, 'az': 0.08006423596040216, 'be': 0.02405042974167787, 'bg': 0.36256966525862144, 'bn': 0.16932799125319595, 'ca': 0.21678128203354266, 'ceb': 0.004630439079666076, 'cs': 0.9173945124991305, 'cy': 0.011894768815091324, 'da': 0.41134009652851106, 'de': 6.088564321318315, 'el': 0.7647814151253215, 'en': 43.84274322288712, 'eo': 0.006246857191850909, 'es': 6.593922868862283, 'et': 0.11600277671642918, 'eu': 0.02316416159631424, 'fa': 0.9061401861983142, 'fi': 0.4104814298183851, 'fil': 0.025997864339004244, 'fr': 5.065605192327664, 'ga': 0.006819026580257076, 'gd': 0.002244214150678123, 'gl': 0.04195702534553548, 'gu': 0.01441063684898361, 'ha': 0.004048232805718929, 'hi': 0.2977135190076239, 'ht': 0.0025912338922100043, 'hu': 0.6317185871878461, 'hy': 0.04319238752210946, 'id': 0.21661546153231062, 'ig': 0.0012332209743252396, 'is': 0.035009905801585396, 'it': 2.9852609968497417, 'iw': 0.19636399296789392, 'ja': 0.9504488872823865, 'jv': 0.002441963100025532, 'ka': 0.04156171703245643, 'kk': 0.03815315724645429, 'km': 0.015435927011184868, 'kn': 0.021372376789538385, 'ko': 0.26804610240226684, 'ku': 0.00444998702973429, 'ky': 0.013365830520452667, 'lb': 0.009489551868240827, 'lo': 0.0033747577859191314, 'lt': 0.20335249793655796, 'lv': 0.11020919642770399, 'mg': 0.002936580819485055, 'mi': 0.0016521382726156783, 'mk': 0.04014427460007402, 'ml': 0.040198418021520084, 'mn': 0.03286870355284287, 'mr': 0.05080447303733959, 'ms': 0.05228142304888498, 'mt': 0.01236980345565089, 'my': 0.013920513423827437, 'ne': 0.052185247381212295, 'nl': 1.5209163321594643, 'no': 0.3417524286083547, 'ny': 0.0012819556296742231, 'pa': 0.008134271213642219, 'pl': 1.9927742538369295, 'ps': 0.005546171060140008, 'pt': 2.7478980104132935, 'ro': 0.7416131941664106, 'ru': 11.30893826638371, 'sd': 0.0023512519589251584, 'si': 0.009436066420402445, 'sk': 0.29799791973547507, 'sl': 0.13808396951496665, 'sm': 0.0011389077065251587, 'sn': 0.00139383344717102, 'so': 0.013026829135258525, 'sq': 0.0783275536552463, 'sr': 0.053253673847049675, 'st': 0.0011148749417020328, 'su': 0.0017096384746704515, 'sv': 0.7060202568358447, 'sw': 0.014268079618016549, 'ta': 0.06434238038434485, 'te': 0.022692595257313836, 'tg': 0.017434114793063466, 'th': 0.3124951414503922, 'tr': 1.4794698831813984, 'uk': 0.6262971188545052, 'ur': 0.038279320897704416, 'uz': 0.013199653152179166, 'vi': 1.4795213835563188, 'xh': 0.0013631428816057104, 'yi': 0.0019350000113506024, 'yo': 0.0009667305111171593, 'zh': 2.3961001426070587, 'zu': 0.0029133621576002537}
mc42nllb = { 'de': 'deu_Latn', 'af': 'afr_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab', 'az': 'azj_Latn', 'be': 'bel_Cyrl', 'bn': 'ben_Beng', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'ceb': 'ceb_Latn', 'cs': 'ces_Latn', 'co': 'NO', 'cy': 'cym_Latn', 'da': 'dan_Latn','el': 'ell_Grek', 'en': 'eng_Latn', 'eo': 'epo_Latn', 'et': 'est_Latn', 'eu': 'eus_Latn', 'fa': 'pes_Arab', 'fil': 'tgl_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn', 'fy': 'NO', 'gd': 'gla_Latn', 'ga': 'gle_Latn', 'gl': 'glg_Latn', 'gu': 'guj_Gujr', 'ht': 'hat_Latn', 'ha': 'hau_Latn', 'haw': 'NO', 'iw': 'heb_Hebr', 'hi': 'hin_Deva', 'hmn': 'NO', 'hu': 'hun_Latn', 'hy': 'hye_Armn', 'ig': 'ibo_Latn', 'id': 'ind_Latn', 'is': 'isl_Latn', 'it': 'ita_Latn', 'jv': 'jav_Latn', 'ja': 'jpn_Jpan', 'kn': 'kan_Knda', 'ka': 'kat_Geor', 'kk': 'kaz_Cyrl', 'km': 'khm_Khmr', 'ky': 'kir_Cyrl', 'ko': 'kor_Hang', 'ku': 'ckb_Arab', 'lo': 'lao_Laoo', 'la': 'NO', 'lv': 'lvs_Latn', 'lt': 'lit_Latn', 'lb': 'ltz_Latn', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'mk': 'mkd_Cyrl', 'mg': 'plt_Latn', 'mt': 'mlt_Latn', 'mn': 'khk_Cyrl', 'mi': 'mri_Latn', 'ms': 'zsm_Latn', 'my': 'mya_Mymr', 'ne': 'npi_Deva', 'nl': 'nld_Latn', 'no': 'nob_Latn', 'ny': 'nya_Latn', 'pa': 'pan_Guru', 'pl': 'pol_Latn', 'pt': 'por_Latn', 'ps': 'pbt_Arab', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'si': 'sin_Sinh', 'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sm': 'smo_Latn', 'sn': 'sna_Latn', 'sd': 'snd_Arab', 'so': 'som_Latn', 'st': 'sot_Latn', 'es': 'spa_Latn', 'sq': 'als_Latn', 'sr': 'srp_Cyrl', 'su': 'sun_Latn', 'sw': 'swh_Latn', 'sv': 'swe_Latn', 'ta': 'tam_Taml', 'te': 'tel_Telu', 'tg': 'tgk_Cyrl', 'th': 'tha_Thai', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vi': 'vie_Latn', 'xh': 'xho_Latn', 'yi': 'ydd_Hebr', 'yo': 'yor_Latn', 'zh': 'zho_Hant', 'zu': 'zul_Latn'}

lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}

seed = 42
np.random.seed(seed)

nllb_model = "facebook/nllb-200-distilled-1.3B"
model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model)
tokenizer = AutoTokenizer.from_pretrained(nllb_model)
model = model.to("cuda")

def collator(examples):
    inputs = tokenizer(examples, return_tensors="pt", padding="longest")
    return inputs

class CaptionDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.captions[index]

def translate(model, tokenizer, nllb_lang, text, batchsize = 256):
    dataloader = DataLoader(CaptionDataset(text), batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collator)
    results = []
    for batch in tqdm(dataloader):
        batch = batch.to(model.device)
        with torch.cuda.amp.autocast():
            translated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id[nllb_lang])
        mt_prompt = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        results.extend(mt_prompt)
    return results


root = "."

file = "detail_23k_en.json"
data = json.load(open(os.path.join(root, file)))
np.random.shuffle(data)
mt_data = []
idx = 0
print(f"Total examples: {len(data)}")
for lang, part in tqdm(mc4_lang_dist.items(), total=len(mc4_lang_dist)):
    num_examples = int(len(data) * part/100)
    examples = data[idx:idx+num_examples]
    idx = idx+num_examples
    print(f"{lang}: {num_examples} ({part}%)")
    if num_examples == 0:
        continue
    if lang == "en":
        pass
    else:
        nllb_lang = mc42nllb[lang]
        captions = [ex["label"] for ex in examples]
        translates_captions = translate(model, tokenizer, nllb_lang, captions, batchsize=64)
        instructions = [ex["context"] for ex in examples]
        translated_instruction = translate(model, tokenizer, nllb_lang, instructions)
        for i, ex in enumerate(examples):
            ex["context"] = translated_instruction[i]
            ex["label"] = translates_captions[i]
        if lang == "de":
            print(examples[:5])
    mt_data.extend(examples)
with open(os.path.join(root, f"detail_23k_mt.json"), "w", encoding="utf-8") as f:
    json.dump(mt_data, f, ensure_ascii=False, indent=2)

file = "conversation_58k_split_max3sent_en.json"
data = json.load(open(os.path.join(root, file)))
np.random.shuffle(data)
mt_data = []
idx = 0
print(f"Total examples: {len(data)}")
for lang, part in tqdm(mc4_lang_dist.items(), total=len(mc4_lang_dist)):
    num_examples = int(len(data) * part/100)
    examples = data[idx:idx+num_examples]
    idx = idx+num_examples
    print(f"{lang}: {num_examples} ({part}%)")
    if num_examples == 0:
        continue
    if lang == "en":
        pass
    else:
        nllb_lang = mc42nllb[lang]
        captions = [ex["label"] for ex in examples]
        translates_captions = translate(model, tokenizer, nllb_lang, captions, batchsize=64)
        instructions = [ex["context"] for ex in examples]
        translated_instruction = translate(model, tokenizer, nllb_lang, instructions)
        for i, ex in enumerate(examples):
            ex["context"] = translated_instruction[i]
            ex["label"] = translates_captions[i]
        if lang == "de":
            print(examples[:5])
    mt_data.extend(examples)
with open(os.path.join(root, f"conversation_58k_split_max3sent_mt.json"), "w", encoding="utf-8") as f:
    json.dump(mt_data, f, ensure_ascii=False, indent=2)

