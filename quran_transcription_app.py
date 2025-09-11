import os
import threading
import time
import difflib
import re
from typing import List, Tuple, Optional, Dict, Set
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

def normalize_arabic_text(text: str) -> str:
    """تطبيع النص العربي بإزالة التشكيل وتوحيد الحروف"""
    if not text:
        return ""
    
    # إزالة التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    # توحيد الألف
    text = re.sub(r'[آأإ]', 'ا', text)
    # توحيد الواو
    text = re.sub(r'[ؤ]', 'و', text)
    # توحيد الياء
    text = re.sub(r'[ئ]', 'ي', text)
    # توحيد التاء المربوطة
    text = re.sub(r'ة', 'ه', text)
    # إزالة الأحرف غير العربية
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # تنظيف المسافات
    text = ' '.join(text.split())
    
    return text.strip().lower()

def is_basmala(text: str) -> bool:
    """فحص إذا كان النص بسملة"""
    normalized = normalize_arabic_text(text)
    basmala_patterns = [
        "بسم الله الرحمن الرحيم",
        "بسم الله الرحمان الرحيم", 
        "باسم الله الرحمن الرحيم"
    ]
    
    for pattern in basmala_patterns:
        if difflib.SequenceMatcher(None, normalized, pattern).ratio() > 0.8:
            return True
    return False

class FastTextMatcher:
    """فئة للبحث السريع في النصوص باستخدام الفهرسة المسبقة"""
    
    def __init__(self, reference_lines: List[str]):
        self.reference_lines = reference_lines
        self.normalized_lines = [normalize_arabic_text(line) for line in reference_lines]
        
        # فهرسة الكلمات للبحث السريع
        self.word_index = defaultdict(set)
        self.build_word_index()
        
        # فهرسة البسملات
        self.basmala_indices = []
        self.surah_starts = []
        self.identify_basmalas()
        
    def build_word_index(self):
        """بناء فهرس الكلمات للبحث السريع"""
        for line_idx, normalized_line in enumerate(self.normalized_lines):
            words = normalized_line.split()
            for word in words:
                if len(word) >= 3:  # فهرسة الكلمات ذات 3 أحرف أو أكثر فقط
                    self.word_index[word].add(line_idx)
                    # فهرسة الكلمات الجزئية أيضاً
                    for i in range(len(word) - 2):
                        substring = word[i:i+3]
                        self.word_index[substring].add(line_idx)
    
    def identify_basmalas(self):
        """تحديد مواقع البسملات في النص"""
        for i, line in enumerate(self.reference_lines):
            if is_basmala(line):
                self.basmala_indices.append(i)
                if i > 0:  # ليس البسملة الأولى
                    self.surah_starts.append(i)
    
    def get_candidate_lines(self, text: str, max_candidates: int = 20) -> Set[int]:
        """الحصول على المرشحين المحتملين للمطابقة بسرعة"""
        normalized_text = normalize_arabic_text(text)
        words = normalized_text.split()
        
        if not words:
            return set()
        
        candidates = set()
        word_scores = {}
        
        # البحث عن الكلمات في الفهرس
        for word in words:
            if len(word) >= 3:
                matching_lines = self.word_index.get(word, set())
                for line_idx in matching_lines:
                    candidates.add(line_idx)
                    word_scores[line_idx] = word_scores.get(line_idx, 0) + 1
        
        # ترتيب المرشحين حسب عدد الكلمات المطابقة
        sorted_candidates = sorted(candidates, 
                                 key=lambda x: word_scores.get(x, 0), 
                                 reverse=True)
        
        return set(sorted_candidates[:max_candidates])
    
    def calculate_similarity_fast(self, text1: str, text2: str) -> float:
        """حساب التشابه بطريقة سريعة"""
        norm1 = normalize_arabic_text(text1)
        norm2 = normalize_arabic_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # التشابه الأساسي
        basic_similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # تشابه الكلمات
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return basic_similarity
        
        # حساب التشابه بناءً على الكلمات المشتركة
        common_words = words1.intersection(words2)
        union_words = words1.union(words2)
        
        word_similarity = len(common_words) / len(union_words) if union_words else 0
        
        # الوزن النهائي
        final_similarity = (basic_similarity * 0.4) + (word_similarity * 0.6)
        
        return min(final_similarity, 1.0)

def detect_reading_pattern_fast(segments: List[Tuple[float, float, str]], 
                               matcher: FastTextMatcher) -> str:
    """كشف نمط القراءة بسرعة"""
    if len(segments) < 3:
        return "sequential"
    
    # فحص العينات الأولى للسرعة
    sample_size = min(5, len(segments))
    sequential_score = 0
    
    for i in range(sample_size):
        seg_text = segments[i][2]
        candidates = matcher.get_candidate_lines(seg_text, 5)
        
        # البحث في المرشحين فقط
        best_pos = -1
        best_score = 0
        
        for candidate in candidates:
            if candidate < 50:  # فحص الجزء الأول من المصحف فقط
                similarity = matcher.calculate_similarity_fast(
                    seg_text, matcher.reference_lines[candidate])
                if similarity > best_score:
                    best_score = similarity
                    best_pos = candidate
        
        if best_pos >= 0 and best_score > 0.5:
            expected_pos = i * 2  # توقع تقريبي
            if abs(best_pos - expected_pos) < 15:
                sequential_score += 1
    
    return "sequential" if sequential_score >= 2 else "nonsequential"

def find_fast_sequential_alignment(segments: List[Tuple[float, float, str]], 
                                 matcher: FastTextMatcher) -> List[Tuple[float, float, str, int]]:
    """محاذاة سريعة للقراءة المتتالية"""
    print(f"بدء المحاذاة المتتالية السريعة لـ {len(segments)} مقطع")
    
    # العثور على نقطة البداية
    start_ref_idx = find_starting_position_fast(segments, matcher)
    print(f"نقطة البداية المحددة: السطر {start_ref_idx + 1}")
    
    aligned_segments = []
    current_ref_idx = start_ref_idx
    seg_idx = 0
    
    while seg_idx < len(segments) and current_ref_idx < len(matcher.reference_lines):
        current_segment = segments[seg_idx]
        seg_text = current_segment[2]
        
        # التعامل مع البسملة
        if is_basmala(seg_text) and current_ref_idx not in matcher.basmala_indices:
            next_basmala = find_next_basmala(current_ref_idx, matcher.basmala_indices)
            if next_basmala:
                current_ref_idx = next_basmala
                print(f"انتقال للبسملة في السطر {current_ref_idx + 1}")
        
        # البحث السريع في نافذة محدودة
        best_match = find_best_match_in_window(
            seg_idx, segments, current_ref_idx, matcher, window_size=10)
        
        if best_match:
            aligned_segments.append(best_match['result'])
            seg_idx += best_match['seg_count']
            current_ref_idx = best_match['new_ref_idx']
            
            print(f"مطابقة المقطع {seg_idx} مع السطر {best_match['new_ref_idx']} "
                  f"(نتيجة: {best_match['score']:.3f})")
        else:
            # لا توجد مطابقة
            aligned_segments.append((
                current_segment[0], current_segment[1],
                seg_text + " [غير مطابق]", -1
            ))
            seg_idx += 1
    
    # إضافة المقاطع المتبقية
    while seg_idx < len(segments):
        segment = segments[seg_idx]
        aligned_segments.append((
            segment[0], segment[1],
            segment[2] + " [متبقي]", -1
        ))
        seg_idx += 1
    
    matched_count = sum(1 for _, _, _, line_num in aligned_segments if line_num > 0)
    print(f"اكتملت المحاذاة: {matched_count}/{len(aligned_segments)} مقطع مطابق")
    
    return aligned_segments

def find_starting_position_fast(segments: List[Tuple[float, float, str]], 
                              matcher: FastTextMatcher) -> int:
    """العثور على نقطة البداية بسرعة"""
    # فحص البسملة في البداية
    if segments and is_basmala(segments[0][2]):
        # التحقق من الفاتحة
        if check_fatiha_pattern(segments, matcher):
            return 0
        
        # البحث عن أفضل بسملة مطابقة
        best_basmala = find_best_basmala_match(segments, matcher)
        if best_basmala is not None:
            return best_basmala
    
    # البحث العام بعينات سريعة
    sample_positions = list(range(0, min(100, len(matcher.reference_lines)), 5))
    best_start = 0
    best_score = 0
    
    for start_pos in sample_positions:
        score = calculate_start_score(segments[:3], start_pos, matcher)
        if score > best_score:
            best_score = score
            best_start = start_pos
    
    return best_start

def check_fatiha_pattern(segments: List[Tuple[float, float, str]], 
                        matcher: FastTextMatcher) -> bool:
    """التحقق من نمط سورة الفاتحة"""
    if len(segments) < 4:
        return False
    
    fatiha_indicators = [
        "الحمد لله رب العالمين",
        "الرحمن الرحيم", 
        "مالك يوم الدين"
    ]
    
    matches = 0
    for i, indicator in enumerate(fatiha_indicators, 1):
        if i < len(segments):
            similarity = matcher.calculate_similarity_fast(
                segments[i][2], indicator)
            if similarity > 0.6:
                matches += 1
    
    return matches >= 2

def find_best_basmala_match(segments: List[Tuple[float, float, str]], 
                           matcher: FastTextMatcher) -> Optional[int]:
    """العثور على أفضل مطابقة للبسملة"""
    if not segments or not matcher.basmala_indices:
        return None
    
    best_basmala = None
    best_score = 0
    
    for basmala_idx in matcher.basmala_indices[1:]:  # تخطي البسملة الأولى
        score = calculate_basmala_context_score(segments, basmala_idx, matcher)
        if score > best_score:
            best_score = score
            best_basmala = basmala_idx
    
    return best_basmala if best_score > 0.6 else None

def calculate_basmala_context_score(segments: List[Tuple[float, float, str]], 
                                  basmala_idx: int, 
                                  matcher: FastTextMatcher) -> float:
    """حساب نتيجة السياق للبسملة"""
    if basmala_idx + 3 >= len(matcher.reference_lines):
        return 0
    
    score = 0
    matches = 0
    
    for i in range(1, min(4, len(segments))):
        if basmala_idx + i < len(matcher.reference_lines):
            similarity = matcher.calculate_similarity_fast(
                segments[i][2], 
                matcher.reference_lines[basmala_idx + i]
            )
            if similarity > 0.5:
                score += similarity
                matches += 1
    
    return score / max(matches, 1)

def calculate_start_score(segments: List[Tuple[float, float, str]], 
                         start_pos: int, 
                         matcher: FastTextMatcher) -> float:
    """حساب نتيجة نقطة البداية"""
    score = 0
    matches = 0
    
    for i, segment in enumerate(segments):
        if start_pos + i >= len(matcher.reference_lines):
            break
        
        similarity = matcher.calculate_similarity_fast(
            segment[2], matcher.reference_lines[start_pos + i])
        
        if similarity > 0.5:
            score += similarity
            matches += 1
    
    return score / max(matches, 1)

def find_next_basmala(current_pos: int, basmala_indices: List[int]) -> Optional[int]:
    """العثور على البسملة التالية"""
    for basmala_idx in basmala_indices:
        if basmala_idx > current_pos:
            return basmala_idx
    return None

def find_best_match_in_window(seg_idx: int, 
                            segments: List[Tuple[float, float, str]], 
                            current_ref_idx: int,
                            matcher: FastTextMatcher,
                            window_size: int = 10) -> Optional[Dict]:
    """البحث عن أفضل مطابقة في نافذة محدودة"""
    current_segment = segments[seg_idx]
    best_match = None
    best_score = 0.4
    
    # تحديد نافذة البحث
    window_start = max(0, current_ref_idx - 2)
    window_end = min(len(matcher.reference_lines), current_ref_idx + window_size)
    
    # البحث في النافذة
    for ref_idx in range(window_start, window_end):
        # مطابقة مفردة
        similarity = matcher.calculate_similarity_fast(
            current_segment[2], matcher.reference_lines[ref_idx])
        
        if similarity > best_score:
            best_score = similarity
            best_match = {
                'result': (current_segment[0], current_segment[1], 
                          matcher.reference_lines[ref_idx], ref_idx + 1),
                'seg_count': 1,
                'new_ref_idx': ref_idx + 1,
                'score': similarity
            }
    
    # مطابقة متعددة المقاطع (للسرعة، نحدد العدد)
    if seg_idx + 1 < len(segments) and best_score < 0.7:
        for seg_count in range(2, min(3, len(segments) - seg_idx + 1)):
            combined_text = " ".join(segments[seg_idx + j][2] for j in range(seg_count))
            
            for ref_idx in range(window_start, min(window_end, len(matcher.reference_lines))):
                similarity = matcher.calculate_similarity_fast(
                    combined_text, matcher.reference_lines[ref_idx])
                
                if similarity > best_score:
                    best_score = similarity
                    start_time = segments[seg_idx][0]
                    end_time = segments[seg_idx + seg_count - 1][1]
                    best_match = {
                        'result': (start_time, end_time,
                                  matcher.reference_lines[ref_idx], ref_idx + 1),
                        'seg_count': seg_count,
                        'new_ref_idx': ref_idx + 1,
                        'score': similarity
                    }
    
    return best_match

def find_fast_nonsequential_alignment(segments: List[Tuple[float, float, str]], 
                                    matcher: FastTextMatcher) -> List[Tuple[float, float, str, int]]:
    """محاذاة سريعة للقراءة غير المتتالية"""
    print(f"بدء المحاذاة غير المتتالية السريعة لـ {len(segments)} مقطع")
    
    aligned_segments = []
    used_ref_indices = set()
    
    for seg_idx, (start_time, end_time, seg_text) in enumerate(segments):
        print(f"معالجة المقطع {seg_idx + 1}: '{seg_text[:30]}...'")
        
        # الحصول على المرشحين بسرعة
        candidates = matcher.get_candidate_lines(seg_text, max_candidates=15)
        
        best_match = None
        best_score = 0.4
        
        # البحث في المرشحين فقط
        for ref_idx in candidates:
            if ref_idx in used_ref_indices:
                continue  # تخطي المستخدمة بالفعل
                
            similarity = matcher.calculate_similarity_fast(
                seg_text, matcher.reference_lines[ref_idx])
            
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    'ref_idx': ref_idx,
                    'score': similarity
                }
        
        if best_match:
            ref_idx = best_match['ref_idx']
            ref_text = matcher.reference_lines[ref_idx]
            
            aligned_segments.append((start_time, end_time, ref_text, ref_idx + 1))
            used_ref_indices.add(ref_idx)
            
            print(f"مطابقة المقطع {seg_idx + 1} مع السطر {ref_idx + 1} "
                  f"(نتيجة: {best_match['score']:.3f})")
        else:
            print(f"لا توجد مطابقة للمقطع {seg_idx + 1}")
            aligned_segments.append((start_time, end_time, seg_text + " [غير مطابق]", -1))
    
    matched_count = sum(1 for _, _, _, line_num in aligned_segments if line_num > 0)
    print(f"اكتملت المحاذاة غير المتتالية: {matched_count}/{len(aligned_segments)} مقطع مطابق")
    
    return aligned_segments

def find_optimized_quran_alignment(transcribed_segments: List[Tuple[float, float, str]], 
                                  reference_lines: List[str]) -> List[Tuple[float, float, str, int]]:
    """الدالة الرئيسية للمحاذاة المحسّنة والسريعة"""
    print(f"بدء المحاذاة المحسّنة مع {len(transcribed_segments)} مقطع و {len(reference_lines)} سطر مرجعي")
    
    # إنشاء محرك البحث السريع
    matcher = FastTextMatcher(reference_lines)
    print(f"تم إنشاء الفهارس - عثر على {len(matcher.basmala_indices)} بسملة")
    
    # كشف نمط القراءة بسرعة
    reading_pattern = detect_reading_pattern_fast(transcribed_segments, matcher)
    print(f"نمط القراءة المكتشف: {reading_pattern}")
    
    # اختيار الخوارزمية المناسبة
    if reading_pattern == "sequential":
        return find_fast_sequential_alignment(transcribed_segments, matcher)
    else:
        return find_fast_nonsequential_alignment(transcribed_segments, matcher)

# باقي الكود يبقى كما هو...
def format_ts(seconds: float) -> str:
    if seconds is None:
        return "00:00:00.000"
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def segments_to_srt(segments: List[Tuple[float, float, str, int]]) -> str:
    lines = []
    for i, (start, end, text, line_num) in enumerate(segments, 1):
        start_srt = format_ts(start).replace(".", ",")
        end_srt = format_ts(end).replace(".", ",")
        lines.append(str(i))
        lines.append(f"{start_srt} --> {end_srt}")
        line_info = f" (السطر {line_num})" if line_num > 0 else ""
        lines.append(text.strip() + line_info)
        lines.append("")
    return "\n".join(lines)

class Transcriber(threading.Thread):
    def __init__(self, audio_path, model_name, lang, beam_size, vad, callback):
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.model_name = model_name
        self.lang = None if lang.lower() == "auto" else lang
        self.beam_size = beam_size
        self.vad = vad
        self.callback = callback
        self._stop = False

    def run(self):
        if not WHISPER_AVAILABLE:
            self.callback({"type": "error", "msg": "faster-whisper غير مثبت"})
            return
        try:
            model_path = os.path.expanduser("~/.cache/whisper")
            model = WhisperModel(self.model_name, device="cpu", compute_type="int8", download_root=model_path)
            
            segments, info = model.transcribe(
                self.audio_path,
                language=self.lang,
                beam_size=self.beam_size,
                vad_filter=self.vad
            )
            
            results = []
            for seg in segments:
                if self._stop:
                    return
                results.append((seg.start, seg.end, seg.text))
                self.callback({"type": "partial", "segment": (seg.start, seg.end, seg.text)})
            
            self.callback({"type": "done", "segments": results, "language": getattr(info, 'language', 'unknown')})
        except Exception as e:
            self.callback({"type": "error", "msg": str(e)})

    def stop(self):
        self._stop = True

class Player:
    def __init__(self):
        self.sound = None
        self.duration_ms = 0
        self.paused = False
        self.start_time = 0
        
    def load(self, path: str):
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.load(path)
                self.sound = True
                pygame.mixer.music.play()
                self.start_time = time.time()
                pygame.mixer.music.pause()
                sound = pygame.mixer.Sound(path)
                self.duration_ms = int(sound.get_length() * 1000)
                self.paused = True
            except Exception as e:
                print(f"خطأ في تحميل الصوت: {e}")
                self.sound = None
                
    def toggle(self):
        if not self.sound or not PYGAME_AVAILABLE:
            return
        if self.paused:
            pygame.mixer.music.unpause()
            self.start_time = time.time() - (self.get_time_ms() / 1000.0)
        else:
            pygame.mixer.music.pause()
        self.paused = not self.paused

    def stop(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
            
    def get_time_ms(self):
        if not self.sound or not PYGAME_AVAILABLE:
            return 0
        if self.paused:
            return pygame.mixer.music.get_pos()
        return int((time.time() - self.start_time) * 1000)

    def set_time_ms(self, ms: int):
        if not self.sound or not PYGAME_AVAILABLE:
            return
        pygame.mixer.music.play(start=ms/1000.0)
        self.start_time = time.time() - (ms/1000.0)
        if self.paused:
            pygame.mixer.music.pause()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("محسّن تطبيق تحويل تلاوة القرآن إلى نص مع ترانسكريبت ومحاذاة تلقائية")
        self.segments = []
        self.aligned_segments = []
        self.reference_lines = []
        self.transcriber = None
        self.player = Player()
        self.auto_processing = False  # متغير لتتبع المعالجة التلقائية

        file_frame = tk.Frame(root)
        file_frame.pack(fill="x", padx=5, pady=2)
        
        audio_frame = tk.Frame(file_frame)
        audio_frame.pack(fill="x")
        tk.Label(audio_frame, text="الصوت:", width=8).pack(side="left")
        self.audio_var = tk.StringVar()
        tk.Entry(audio_frame, textvariable=self.audio_var, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(audio_frame, text="استعراض", command=self.browse_audio).pack(side="left", padx=2)

        text_frame = tk.Frame(file_frame)
        text_frame.pack(fill="x")
        tk.Label(text_frame, text="النص المرجعي:", width=8).pack(side="left")
        self.text_var = tk.StringVar()
        tk.Entry(text_frame, textvariable=self.text_var, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(text_frame, text="استعراض", command=self.browse_text).pack(side="left", padx=2)

        opt = tk.Frame(root)
        opt.pack(fill="x", padx=5)
        tk.Label(opt, text="النموذج:").pack(side="left")
        self.model_var = tk.StringVar(value="small")
        ttk.Combobox(opt, textvariable=self.model_var, values=["tiny","base","small","medium","large-v3"], width=10).pack(side="left", padx=2)
        
        tk.Label(opt, text="اللغة:").pack(side="left")
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(opt, textvariable=self.lang_var, values=["auto","ar","en","fr","de","es"], width=8).pack(side="left", padx=2)

        # الأزرار الاختيارية (للاستخدام اليدوي)
        tk.Button(opt, text="تحويل إلى نص (يدوي)", command=self.start_transcribe, bg="#4CAF50", fg="white").pack(side="left", padx=5)
        tk.Button(opt, text="محاذاة سريعة (يدوية)", command=self.align_text, bg="#2196F3", fg="white").pack(side="left", padx=2)
        
        self.play_button = tk.Button(opt, text="تشغيل/إيقاف", command=self.play_toggle, 
                                   state="disabled" if not PYGAME_AVAILABLE else "normal")
        self.play_button.pack(side="left", padx=2)
        
        tk.Button(opt, text="تصدير SRT", command=self.export_srt).pack(side="left", padx=2)

        # إضافة مربع تحديد للمعالجة التلقائية
        auto_frame = tk.Frame(opt)
        auto_frame.pack(side="left", padx=10)
        self.auto_process_var = tk.BooleanVar(value=True)
        tk.Checkbutton(auto_frame, text="معالجة تلقائية", variable=self.auto_process_var,
                      font=("Arial", 10, "bold"), fg="green").pack()

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        orig_frame = ttk.Frame(notebook)
        notebook.add(orig_frame, text="النص المحول الأصلي")
        
        self.orig_tree = ttk.Treeview(orig_frame, columns=("start","end","text"), show="headings")
        self.orig_tree.heading("start", text="البداية")
        self.orig_tree.heading("end", text="النهاية")
        self.orig_tree.heading("text", text="النص المحول")
        self.orig_tree.column("start", width=100)
        self.orig_tree.column("end", width=100)
        self.orig_tree.pack(fill="both", expand=True)
        
        aligned_frame = ttk.Frame(notebook)
        notebook.add(aligned_frame, text="المحاذاة السريعة مع النص المرجعي")
        
        self.aligned_tree = ttk.Treeview(aligned_frame, columns=("start","end","line","text"), show="headings")
        self.aligned_tree.heading("start", text="البداية")
        self.aligned_tree.heading("end", text="النهاية") 
        self.aligned_tree.heading("line", text="رقم السطر")
        self.aligned_tree.heading("text", text="النص المرجعي (القرآن)")
        self.aligned_tree.column("start", width=100)
        self.aligned_tree.column("end", width=100)
        self.aligned_tree.column("line", width=80)
        self.aligned_tree.pack(fill="both", expand=True)

        status_frame = tk.Frame(root)
        status_frame.pack(fill="x")
        self.status = tk.Label(status_frame, text="جاهز - قم بتحميل ملف الصوت والنص المرجعي للقرآن (المعالجة التلقائية مفعلة)", anchor="w", bg="lightgray")
        self.status.pack(fill="x")
        
        self.ref_status = tk.Label(status_frame, text="لم يتم تحميل النص المرجعي", anchor="w", fg="blue")
        self.ref_status.pack(fill="x")

        if not WHISPER_AVAILABLE:
            self.status.config(text="تحذير: faster-whisper غير مثبت", fg="red")
        if not PYGAME_AVAILABLE:
            self.ref_status.config(text="تحذير: pygame غير متاح لتشغيل الصوت", fg="orange")

        self.update_loop()

    def browse_audio(self):
        path = filedialog.askopenfilename(
            title="اختيار ملف صوتي للقرآن",
            filetypes=[("ملفات الصوت","*.mp3 *.wav *.m4a *.flac")]
        )
        if path:
            self.audio_var.set(path)
            self.player.load(path)
            self.status.config(text=f"تم تحميل الصوت: {os.path.basename(path)}")
            
            # بدء المعالجة التلقائية إذا كانت مفعلة
            if self.auto_process_var.get():
                self.auto_processing = True
                self.status.config(text=f"تم تحميل الصوت: {os.path.basename(path)} - بدء المعالجة التلقائية...")
                # بدء الترانسكريبت تلقائياً بعد تأخير قصير
                self.root.after(500, self.start_auto_transcribe)

    def browse_text(self):
        path = filedialog.askopenfilename(
            title="اختيار ملف النص المرجعي للقرآن",
            filetypes=[("ملفات النص","*.txt"), ("جميع الملفات","*.*")]
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.reference_lines = [line.strip() for line in f if line.strip()]
                self.text_var.set(path)
                
                basmala_count = sum(1 for line in self.reference_lines if is_basmala(line))
                self.ref_status.config(text=f"تم تحميل النص المرجعي: {len(self.reference_lines)} آية، "
                                           f"{basmala_count} بسملة مكتشفة")
                
                # إذا كان هناك ملف صوتي محمل وترانسكريبت جاهز، ابدأ المحاذاة
                if self.auto_processing and self.segments and self.auto_process_var.get():
                    self.root.after(500, self.start_auto_alignment)
                    
            except Exception as e:
                messagebox.showerror("خطأ", f"لا يمكن تحميل ملف النص المرجعي:\n{str(e)}")

    def start_auto_transcribe(self):
        """بدء الترانسكريبت التلقائي"""
        if not WHISPER_AVAILABLE:
            messagebox.showerror("خطأ", "faster-whisper غير مثبت. قم بتثبيته بـ:\npip install faster-whisper")
            self.auto_processing = False
            return
            
        audio_path = self.audio_var.get()
        if not audio_path:
            self.auto_processing = False
            return
            
        self.orig_tree.delete(*self.orig_tree.get_children())
        self.aligned_tree.delete(*self.aligned_tree.get_children())
        self.status.config(text="المعالجة التلقائية: جاري تحويل الصوت إلى نص...")

        def callback(msg):
            self.root.after(0, self.handle_auto_msg, msg)

        self.transcriber = Transcriber(
            audio_path=audio_path,
            model_name=self.model_var.get(),
            lang=self.lang_var.get(),
            beam_size=5,
            vad=True,
            callback=callback,
        )
        self.transcriber.start()

    def start_transcribe(self):
        """الترانسكريبت اليدوي"""
        self.auto_processing = False  # إيقاف المعالجة التلقائية عند الاستخدام اليدوي
        
        if not WHISPER_AVAILABLE:
            messagebox.showerror("خطأ", "faster-whisper غير مثبت. قم بتثبيته بـ:\npip install faster-whisper")
            return
            
        audio_path = self.audio_var.get()
        if not audio_path:
            messagebox.showerror("خطأ", "يرجى اختيار ملف صوتي للقرآن")
            return
            
        self.orig_tree.delete(*self.orig_tree.get_children())
        self.aligned_tree.delete(*self.aligned_tree.get_children())
        self.status.config(text="جاري تحويل الصوت إلى نص (يدوي)...")

        def callback(msg):
            self.root.after(0, self.handle_msg, msg)

        self.transcriber = Transcriber(
            audio_path=audio_path,
            model_name=self.model_var.get(),
            lang=self.lang_var.get(),
            beam_size=5,
            vad=True,
            callback=callback,
        )
        self.transcriber.start()

    def handle_auto_msg(self, msg):
        """التعامل مع رسائل المعالجة التلقائية"""
        if msg["type"] == "partial":
            s, e, t = msg["segment"]
            self.orig_tree.insert("", "end", values=(format_ts(s), format_ts(e), t.strip()))
        elif msg["type"] == "done":
            self.segments = msg["segments"]
            self.status.config(text=f"المعالجة التلقائية: اكتمل التحويل — اللغة: {msg['language']} — {len(self.segments)} مقطع")
            
            # بدء المحاذاة التلقائية إذا كان النص المرجعي محملاً
            if self.reference_lines and self.auto_process_var.get():
                self.root.after(1000, self.start_auto_alignment)  # تأخير أطول قبل المحاذاة
            else:
                self.status.config(text=self.status.cget("text") + " — في انتظار النص المرجعي للمحاذاة التلقائية")
                self.auto_processing = True  # الحفاظ على حالة المعالجة التلقائية
                
        elif msg["type"] == "error":
            messagebox.showerror("خطأ", msg["msg"])
            self.status.config(text="خطأ أثناء التحويل التلقائي")
            self.auto_processing = False

    def handle_msg(self, msg):
        """التعامل مع رسائل المعالجة اليدوية"""
        if msg["type"] == "partial":
            s, e, t = msg["segment"]
            self.orig_tree.insert("", "end", values=(format_ts(s), format_ts(e), t.strip()))
        elif msg["type"] == "done":
            self.segments = msg["segments"]
            self.status.config(text=f"اكتمل التحويل — اللغة: {msg['language']} — {len(self.segments)} مقطع")
            if self.reference_lines:
                self.status.config(text=self.status.cget("text") + " — جاهز للمحاذاة السريعة")
        elif msg["type"] == "error":
            messagebox.showerror("خطأ", msg["msg"])
            self.status.config(text="خطأ أثناء التحويل")

    def start_auto_alignment(self):
        """بدء المحاذاة التلقائية"""
        if not self.segments:
            self.auto_processing = False
            return
        if not self.reference_lines:
            self.auto_processing = False
            return

        self.status.config(text="المعالجة التلقائية: جاري المحاذاة السريعة مع النص المرجعي...")
        
        def align_worker():
            try:
                start_time = time.time()
                aligned = find_optimized_quran_alignment(self.segments, self.reference_lines)
                end_time = time.time()
                alignment_time = end_time - start_time
                
                def show_results():
                    self.show_auto_aligned_results(aligned, alignment_time)
                
                self.root.after(0, show_results)
            except Exception as error:
                error_msg = str(error)
                def show_error():
                    messagebox.showerror("خطأ", f"فشلت المحاذاة التلقائية:\n{error_msg}")
                    self.status.config(text="فشلت المحاذاة التلقائية")
                    self.auto_processing = False
                self.root.after(0, show_error)
        
        threading.Thread(target=align_worker, daemon=True).start()

    def align_text(self):
        """المحاذاة اليدوية"""
        if not self.segments:
            messagebox.showerror("خطأ", "يرجى تحويل الصوت إلى نص أولاً")
            return
        if not self.reference_lines:
            messagebox.showerror("خطأ", "يرجى تحميل ملف النص المرجعي أولاً")
            return

        self.status.config(text="جاري المحاذاة السريعة مع النص المرجعي (يدوية)...")
        
        def align_worker():
            try:
                start_time = time.time()
                aligned = find_optimized_quran_alignment(self.segments, self.reference_lines)
                end_time = time.time()
                alignment_time = end_time - start_time
                
                def show_results():
                    self.show_aligned_results(aligned, alignment_time)
                
                self.root.after(0, show_results)
            except Exception as error:
                error_msg = str(error)
                def show_error():
                    messagebox.showerror("خطأ", f"فشلت المحاذاة السريعة:\n{error_msg}")
                    self.status.config(text="فشلت المحاذاة السريعة")
                self.root.after(0, show_error)
        
        threading.Thread(target=align_worker, daemon=True).start()

    def show_auto_aligned_results(self, aligned_segments, alignment_time):
        """عرض نتائج المحاذاة التلقائية"""
        self.aligned_segments = aligned_segments
        self.aligned_tree.delete(*self.aligned_tree.get_children())
        
        matched_count = 0
        unmatched_count = 0
        
        for start, end, text, line_num in aligned_segments:
            if line_num > 0:
                matched_count += 1
                line_display = str(line_num)
                if is_basmala(text):
                    self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), f"{line_display} (ب)", text.strip()))
                else:
                    self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), line_display, text.strip()))
            else:
                unmatched_count += 1
                self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), "❌", text.strip()))
        
        accuracy = (matched_count / len(aligned_segments)) * 100 if aligned_segments else 0
        self.status.config(text=f"✅ المعالجة التلقائية مكتملة: {matched_count}/{len(aligned_segments)} مقطع مطابق "
                              f"({accuracy:.1f}% دقة) في {alignment_time:.2f} ثانية")
        self.auto_processing = False  # إنهاء المعالجة التلقائية

    def show_aligned_results(self, aligned_segments, alignment_time):
        """عرض نتائج المحاذاة اليدوية"""
        self.aligned_segments = aligned_segments
        self.aligned_tree.delete(*self.aligned_tree.get_children())
        
        matched_count = 0
        unmatched_count = 0
        
        for start, end, text, line_num in aligned_segments:
            if line_num > 0:
                matched_count += 1
                line_display = str(line_num)
                if is_basmala(text):
                    self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), f"{line_display} (ب)", text.strip()))
                else:
                    self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), line_display, text.strip()))
            else:
                unmatched_count += 1
                self.aligned_tree.insert("", "end", values=(format_ts(start), format_ts(end), "❌", text.strip()))
        
        accuracy = (matched_count / len(aligned_segments)) * 100 if aligned_segments else 0
        self.status.config(text=f"اكتملت المحاذاة السريعة: {matched_count}/{len(aligned_segments)} مقطع مطابق "
                              f"({accuracy:.1f}% دقة) في {alignment_time:.2f} ثانية")

    def play_toggle(self):
        self.player.toggle()

    def export_srt(self):
        if not self.aligned_segments:
            messagebox.showerror("خطأ", "لا توجد مقاطع محاذاة للتصدير")
            return
            
        path = filedialog.asksaveasfilename(
            title="حفظ ملف SRT للقرآن",
            defaultextension=".srt",
            filetypes=[("ملفات SRT", "*.srt"), ("جميع الملفات", "*.*")]
        )
        
        if path:
            try:
                srt_content = segments_to_srt(self.aligned_segments)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                messagebox.showinfo("نجح", f"تم حفظ ملف SRT للقرآن في:\n{path}")
            except Exception as e:
                messagebox.showerror("خطأ", f"لا يمكن حفظ ملف SRT:\n{str(e)}")

    def update_loop(self):
        current_segments = self.aligned_segments if self.aligned_segments else self.segments
        current_tree = self.aligned_tree if self.aligned_segments else self.orig_tree
        
        if current_segments and self.player.duration_ms > 0:
            cur_time = self.player.get_time_ms() / 1000.0
            for i, segment in enumerate(current_segments):
                start_time = segment[0]
                end_time = segment[1]
                if start_time <= cur_time < end_time:
                    children = current_tree.get_children()
                    if i < len(children):
                        current_tree.selection_set(children[i])
                        current_tree.see(children[i])
                    break
        
        self.root.after(200, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x900")
    app = App(root)
    root.mainloop()