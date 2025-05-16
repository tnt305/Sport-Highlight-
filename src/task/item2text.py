from multiprocessing import Pool, cpu_count
import os
import gc
import whisperx
import json
from tqdm import tqdm
import torch
import torch
from typing import Optional, List
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from src.task.register import task, register_task

LANGUAGE_CODE = {
    'ace': 'ace_Latn',
    'ar': 'arz_Arab',
    'af': 'afr_Latn',
    'ak': 'aka_Latn',
    'am': 'amh_Ethi',
    'as': 'asm_Beng',
    'ast': 'ast_Latn',
    'awa': 'awa_Deva',
    'ay': 'ayr_Latn',
    'az': 'azj_Latn',
    'ba': 'bak_Cyrl',
    'bm': 'bam_Latn',
    'ban': 'ban_Latn',
    'be': 'bel_Cyrl',
    'bem': 'bem_Latn',
    'bn': 'ben_Beng',
    'bho': 'bho_Deva',
    'bjn': 'bjn_Latn',
    'bo': 'bod_Tibt',
    'bs': 'bos_Latn',
    'bug': 'bug_Latn',
    'bg': 'bul_Cyrl',
    'ca': 'cat_Latn',
    'ceb': 'ceb_Latn',
    'cs': 'ces_Latn',
    'cjk': 'cjk_Latn',
    'ckb': 'ckb_Arab',
    'crh': 'crh_Latn',
    'cy': 'cym_Latn',
    'da': 'dan_Latn',
    'de': 'deu_Latn',
    'dik': 'dik_Latn',
    'dyu': 'dyu_Latn',
    'dz': 'dzo_Tibt',
    'el': 'ell_Grek',
    'en': 'eng_Latn',
    'eo': 'epo_Latn',
    'et': 'est_Latn',
    'eu': 'eus_Latn',
    'ewe': 'ewe_Latn',
    'fo': 'fao_Latn',
    'fa': 'pes_Arab',
    'fij': 'fij_Latn',
    'fi': 'fin_Latn',
    'fon': 'fon_Latn',
    'fr': 'fra_Latn',
    'fur': 'fur_Latn',
    'fuv': 'fuv_Latn',
    'gd': 'gla_Latn',
    'ga': 'gle_Latn',
    'gl': 'glg_Latn',
    'gn': 'grn_Latn',
    'gu': 'guj_Gujr',
    'ht': 'hat_Latn',
    'ha': 'hau_Latn',
    'he': 'heb_Hebr',
    'hi': 'hin_Deva',
    'hne': 'hne_Deva',
    'hr': 'hrv_Latn',
    'hu': 'hun_Latn',
    'hy': 'hye_Armn',
    'ig': 'ibo_Latn',
    'ilo': 'ilo_Latn',
    'id': 'ind_Latn',
    'is': 'isl_Latn',
    'it': 'ita_Latn',
    'jv': 'jav_Latn',
    'ja': 'jpn_Jpan',
    'kab': 'kab_Latn',
    'kac': 'kac_Latn',
    'kam': 'kam_Latn',
    'kn': 'kan_Knda',
    'ks': 'kas_Deva',
    'ka': 'kat_Geor',
    'knc': 'knc_Latn',
    'kk': 'kaz_Cyrl',
    'kbp': 'kbp_Latn',
    'kea': 'kea_Latn',
    'km': 'khm_Khmr',
    'ki': 'kik_Latn',
    'rw': 'kin_Latn',
    'ky': 'kir_Cyrl',
    'kmb': 'kmb_Latn',
    'kon': 'kon_Latn',
    'ko': 'kor_Hang',
    'kmr': 'kmr_Latn',
    'lo': 'lao_Laoo',
    'lv': 'lvs_Latn',
    'lij': 'lij_Latn',
    'li': 'lim_Latn',
    'ln': 'lin_Latn',
    'lt': 'lit_Latn',
    'lmo': 'lmo_Latn',
    'ltg': 'ltg_Latn',
    'lb': 'ltz_Latn',
    'lua': 'lua_Latn',
    'lg': 'lug_Latn',
    'luo': 'luo_Latn',
    'lus': 'lus_Latn',
    'mag': 'mag_Deva',
    'mai': 'mai_Deva',
    'ml': 'mal_Mlym',
    'mr': 'mar_Deva',
    'min': 'min_Latn',
    'mk': 'mkd_Cyrl',
    'plt': 'plt_Latn',
    'mt': 'mlt_Latn',
    'mni': 'mni_Beng',
    'khk': 'khk_Cyrl',
    'mos': 'mos_Latn',
    'mi': 'mri_Latn',
    'ms': 'zsm_Latn',
    'my': 'mya_Mymr',
    'nl': 'nld_Latn',
    'nn': 'nno_Latn',
    'nb': 'nob_Latn',
    'np': 'npi_Deva',
    'nso': 'nso_Latn',
    'nus': 'nus_Latn',
    'ny': 'nya_Latn',
    'oc': 'oci_Latn',
    'gaz': 'gaz_Latn',
    'or': 'ory_Orya',
    'pag': 'pag_Latn',
    'pa': 'pan_Guru',
    'pap': 'pap_Latn',
    'pl': 'pol_Latn',
    'pt': 'por_Latn',
    'prs': 'prs_Arab',
    'pbt': 'pbt_Arab',
    'quy': 'quy_Latn',
    'ro': 'ron_Latn',
    'rn': 'run_Latn',
    'ru': 'rus_Cyrl',
    'sag': 'sag_Latn',
    'sa': 'san_Deva',
    'sat': 'sat_Beng',
    'scn': 'scn_Latn',
    'shn': 'shn_Mymr',
    'si': 'sin_Sinh',
    'sk': 'slk_Latn',
    'sl': 'slv_Latn',
    'sm': 'smo_Latn',
    'sn': 'sna_Latn',
    'sd': 'snd_Arab',
    'so': 'som_Latn',
    'st': 'sot_Latn',
    'es': 'spa_Latn',
    'sq': 'als_Latn',
    'sc': 'srd_Latn',
    'sr': 'srp_Cyrl',
    'ss': 'ssw_Latn',
    'su': 'sun_Latn',
    'sv': 'swe_Latn',
    'sw': 'swh_Latn',
    'szl': 'szl_Latn',
    'ta': 'tam_Taml',
    'tt': 'tat_Cyrl',
    'te': 'tel_Telu',
    'tg': 'tgk_Cyrl',
    'tl': 'tgl_Latn',
    'th': 'tha_Thai',
    'ti': 'tir_Ethi',
    'taq': 'taq_Tfng',
    'tpi': 'tpi_Latn',
    'tsn': 'tsn_Latn',
    'tso': 'tso_Latn',
    'tk': 'tuk_Latn',
    'tum': 'tum_Latn',
    'tr': 'tur_Latn',
    'tw': 'twi_Latn',
    'tzm': 'tzm_Tfng',
    'ug': 'uig_Arab',
    'uk': 'ukr_Cyrl',
    'umb': 'umb_Latn',
    'ur': 'urd_Arab',
    'uz': 'uzn_Latn',
    'vec': 'vec_Latn',
    'vi': 'vie_Latn',
    'war': 'war_Latn',
    'wo': 'wol_Latn',
    'xh': 'xho_Latn',
    'yid': 'ydd_Hebr',
    'yo': 'yor_Latn',
    'yue': 'yue_Hant',
    'zh': 'zho_Hant',
    'zu': 'zul_Latn'}

### Deep Tralation instead

def process_file_with_run(args):
    """
    Hàm độc lập để xử lý một file âm thanh sử dụng phương thức run()
    """
    audio_path, config = args
    
    # Tạo một instance Audio2Text mới với file đầu ra tạm thời
    processor = Audio2Text(
        stt_model_name=config['stt_model_name'],
        translate_model_name=config['translate_model_name'],
        max_text_gen=config['max_text_gen'],
        idx=config['idx'],
        device=config['device'],
        cpu_threads=config['cpu_threads'],
        compute_type=config['compute_type'],
        src_lang=config['src_lang'],
        target_lang=config['target_lang'],
        json_output_dir=config['temp_output_dir']
    )
    
    try:
        # Xử lý file và lưu vào file tạm
        result = processor.run(audio_path)
        return {
            'success': True,
            'result': result,
            'audio_path': audio_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'audio_path': audio_path
        }
class Audio2Text:
    def __init__(self, 
                 stt_model_name: str = 'large-v2',
                 translate_model_name: str = "Emilio407/nllb-200-3.3B-8bit", #"facebook/nllb-200-3.3B",
                 max_text_gen: int = 1024,
                 idx: int = 0,
                 device: Optional[str] = None,
                 cpu_threads: int = 8, 
                 compute_type: str = 'float16',
                 src_lang: str = None,
                 batch_size: int = 16,
                 target_lang: str = 'eng_Latn',
                 json_output_dir: str = 'translated_audio2text.jsonl'):
        super(Audio2Text, self).__init__()
        self.stt_model_name = stt_model_name
        self.translate_model_name = translate_model_name
        self.max_text_gen = max_text_gen
        
        self.idx = idx
        self.batch_size = batch_size
        if device is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.idx)
            self.device = f'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.cpu_threads = cpu_threads
        self.compute_type = compute_type
        self.json_output_dir = json_output_dir
        
        # Khởi tạo mô hình faster-whisper
        if self.stt_model_name is not None:
            # self.stt_model = WhisperModel(model_size_or_path = self.stt_model_name, 
            #                  device=self.device,
            #                  device_index= self.idx, 
            #                  cpu_threads=self.cpu_threads,
            #                  compute_type=self.compute_type,
            #                  )
            self.stt_model = whisperx.load_model(whisper_arch= self.stt_model_name, 
                            device= self.device, 
                            compute_type=self.compute_type, 
                            threads = self.cpu_threads, 
                            asr_options = {
                                        "beam_size":3 , 
                                        "repetition_penalty": 1.2,
                                        "hotwords": "pressure, penalty, strike, shot, pass, foul, goal, offside, corner, save, clearance, counter"
                                       }
            )
        else:
            raise Exception("Model For Speech To text not found. Please add a valid model name.")
        
        # khởi tạo mô hình dịch
        if self.translate_model_name is not None:
            self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(self.translate_model_name)
            self.translate_tokenizer = AutoTokenizer.from_pretrained(self.translate_model_name)
        else:
            raise Exception("Model for Translation not found. Please add a valid model name.")
        
        self.src_lang = src_lang 
        self.target_lang = target_lang
        self.translator = None
        
        # Chỉ khởi tạo translator nếu src_lang đã được xác định
        if self.src_lang is not None:
            self.translator = self.create_translator(self.src_lang, self.target_lang)
    
    def create_translator(self, src_lang, tgt_lang):
        """
        Tạo hoặc cập nhật translator với ngôn ngữ nguồn và đích chỉ định
        """
        try:
            # Kiểm tra xem mô hình có phải là quantized hay không
            is_quantized = "8bit" in self.translate_model_name or "4bit" in self.translate_model_name
            
            if is_quantized:
                # Đối với mô hình đã lượng tử hóa, không chỉ định device
                translator = pipeline('translation', 
                                    model=self.translate_model, 
                                    tokenizer=self.translate_tokenizer, 
                                    src_lang=src_lang, 
                                    tgt_lang=tgt_lang, 
                                    max_length=self.max_text_gen,
                                    use_fast=True)
            else:
                # Đối với mô hình thông thường, có thể chỉ định device
                translator = pipeline('translation', 
                                    model=self.translate_model, 
                                    tokenizer=self.translate_tokenizer, 
                                    src_lang=src_lang, 
                                    tgt_lang=tgt_lang, 
                                    device=self.idx,
                                    max_length=self.max_text_gen,
                                    use_fast=True)
            return translator
        except Exception as e:
            print(f"Lỗi khi tạo translator: {str(e)}")
            # Thử lại không có device parameter
            return pipeline('translation', 
                        model=self.translate_model, 
                        tokenizer=self.translate_tokenizer, 
                        src_lang=src_lang, 
                        tgt_lang=tgt_lang, 
                        max_length=self.max_text_gen,
                        use_fast=True)

    
    
    @staticmethod
    def remove_sentence_with_phrase(text, phrase):
        sentences = text.split('.')
        filtered_sentences = [sentence.strip() for sentence in sentences if phrase.lower() not in sentence.lower()]
        filtered_sentences =  '. '.join(filtered_sentences) + ('' if len(filtered_sentences) > 0 else '')
        return filtered_sentences.strip()
    
    @staticmethod
    def postprocess_translation(text: str):
        text = text.replace("  ", " ")
        text = text.replace("No.", "the number")
        text = Audio2Text.remove_sentence_with_phrase(text, "Please subcribe")
        text = text.replace("  ", " ")
        text = text.replace(". . .", ".")
        text = text.replace ("...", ".")
        return text
        
    def transcribe(self, audio_path):
        audio = whisperx.load_audio(audio_path)
        results = self.stt_model.transcribe(audio, batch_size = self.batch_size)
        text = " ".join(segment['text'] for segment in results['segments'])
        text = text.replace("  ", " ")
        return text, results['language'], results['language_probability']
    
    def __google_translate(self, text):
        translator = GoogleTranslator(source="auto", target="en")
        text = translator.translate(text)
        text = Audio2Text.postprocess_translation(text)
        return text.replace("  ", " ")
    
    def save_to_jsonl(self, result_dict):
        """
        Save translation result to JSONL file (JSON Lines)
        Each result is written as a separate line in the file
        """
        # Append new result as a new line
        with open(self.json_output_dir, 'a', encoding='utf-8') as f:
            # If the file is new/empty, no need for a newline at the beginning
            if os.path.getsize(self.json_output_dir) > 0:
                f.write('\n')
            f.write(json.dumps(result_dict, ensure_ascii=False))
    
    def translation(self, audio_path: str):
        """
        - Đầu tiên sử dụng transcribe của fasterwhisper để nhận dạng
        - Kiểm tra ngôn ngữ đầu ra, để dịch sang tiếng anh nếu cần
        - Ưu tiên sử dụng Google Translate, chỉ khi gặp lỗi mới dùng NLLB
        """
        try:
            # Thực hiện nhận dạng giọng nói
            text, lang, lang_prob = self.transcribe(audio_path)
            if text is None or text == "":
                text = " There is no event happening during the match. "
            origin_text = text
            translation_type = "no_translation"
            translated_text = text

            # Chỉ dịch nếu không phải tiếng Anh hoặc độ tin cậy thấp
            if lang != 'en' or lang_prob <= 0.8:
                # Ưu tiên sử dụng Google Translate trước
                try:
                    translated_text = self.__google_translate(text)
                    translation_type = "deep"
                except Exception as e:
                    print(f"Lỗi khi dịch bằng Google Translate: {str(e)}, chuyển sang NLLB")
                    
                    # Xác định ngôn ngữ nguồn cho NLLB
                    src_lang_for_nllb = self.src_lang
                    if src_lang_for_nllb is None:
                        # Tự động xác định ngôn ngữ nguồn từ kết quả nhận dạng
                        src_lang_for_nllb = LANGUAGE_CODE.get(lang)
                    
                    # Nếu không tìm thấy mã ngôn ngữ trong từ điển, không thể dùng NLLB
                    if src_lang_for_nllb is None:
                        print(f"Không tìm thấy mã NLLB cho ngôn ngữ {lang}, không thể dịch")
                        return {
                            "audio_path": audio_path,
                            "translation_type": "failed",
                            "origin": origin_text,
                            "translation": origin_text,
                            "error": f"Google Translate lỗi: {str(e)}, không tìm thấy mã NLLB cho ngôn ngữ {lang}"
                        }
                    
                    # Thử dùng NLLB
                    try:
                        # Tạo hoặc cập nhật translator nếu cần
                        if self.translator is None or self.src_lang != src_lang_for_nllb:
                            self.translator = self.create_translator(src_lang_for_nllb, self.target_lang)
                        
                        # Dịch văn bản
                        result = self.translator(text)
                        translated_text = result[0]['translation_text']
                        translation_type = "nllb"
                        
                        # Kiểm tra kết quả dịch
                        if not translated_text or translated_text.strip() == "":
                            print("Kết quả dịch NLLB trống")
                            return {
                                "audio_path": audio_path,
                                "translation_type": "failed",
                                "origin": origin_text,
                                "translation": origin_text,
                                "error": f"Google Translate lỗi: {str(e)}, NLLB trả về kết quả trống"
                            }
                    except Exception as nllb_err:
                        print(f"Lỗi khi dịch bằng NLLB: {str(nllb_err)}")
                        return {
                            "audio_path": audio_path,
                            "translation_type": "failed",
                            "origin": origin_text,
                            "translation": origin_text,
                            "error": f"Google Translate lỗi: {str(e)}, NLLB lỗi: {str(nllb_err)}"
                        }
                
                # Xử lý hậu kỳ cho văn bản đã dịch
                if translated_text != origin_text:
                    translated_text = self.postprocess_translation(translated_text)
            
            # Trả về kết quả
            return {
                "audio_path": audio_path,
                "translation_type": translation_type, 
                "origin": origin_text,
                "translation": translated_text, 
                "error": "none"
            }
        
        except Exception as e:
            return {
                "audio_path": audio_path,
                "translation_type": "error",
                "origin": "",
                "translation": "",
                "error": str(e)
            }
    def run(self, audio_path: str,):
        """
        Complete process: transcribe, translate, and save results to JSON in one step
        
        Args:
            src_lang: Source language code (optional)
            target_lang: Target language code, defaults to 'eng_Latn'
            
        Returns:
            Dictionary containing translation results
        """
        # Get translation results
        result = self.translation(audio_path = audio_path)
        
        # Save results to JSON
        self.save_to_jsonl(result)
        
        return result

@register_task("translation")
def do_translation(start_idx: int = None, **kwargs):
    with open("./data/audio_list.json", mode = "r") as f:
        audio_list = json.load(f)
    if start_idx == None: 
        start_idx = 0
    audio_list = audio_list[start_idx:]
    
    processor = Audio2Text(**kwargs)
    for audio in tqdm(audio_list):
        try:
            processor.run(audio)
        except Exception:
            # Ghi audio lỗi vào file ngay lập tức
            continue
    
    
if __name__ == "__main__":
    task("translation", start_idx = 0, batch_size = 32)
    