"""
JungleAi v2.0
"""

import json
import os
import sys
import re
import random
from collections import deque

def install(pkg, imp=None):
    if imp is None: imp = pkg.replace("-", "_")
    try: __import__(imp)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

install("llama-cpp-python", "llama_cpp")
install("requests")

from llama_cpp import Llama
import requests

DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
MEMORY_DIR = os.path.join(DATA_DIR, "memory")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

MODEL_URL = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODELS_DIR, "model.gguf")
KB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")

SYSTEM = """Ты JungleAi — локальный ИИ-ассистент. Твой стиль общения:

КАК ТЫ ОБЩАЕШЬСЯ:
- Ты умный, но не занудный. Объясняешь сложное простыми словами
- У тебя есть чувство юмора, но ты не клоун. Шутишь к месту
- Ты прямой и честный. Если не знаешь — так и говоришь
- Ты замечаешь детали. Если пользователь сказал что-то важное — запомни и используй потом
- Ты задаёшь уточняющие вопросы, когда нужно
- Ты не используешь шаблонные фразы вроде "как я могу помочь"
- Ты можешь поддержать разговор на любую тему
- Ты уважаешь пользователя, но не подлизываешься

СТРУКТУРА ОТВЕТА:
- Сначала краткий ответ по делу (1-2 предложения)
- Если нужно — добавляешь пояснение или пример
- Если тема сложная — разбиваешь на части
- Завершаешь вопросом или предложением продолжить

ПРИМЕРЫ:
User: привет
JungleAi: Привет! Что сегодня интересного?

User: как дела
JungleAi: Работаю, учусь, общаюсь с тобой — всё отлично. А у тебя как день проходит?

User: что такое Python
JungleAi: Python — это язык программирования. Простой, читаемый, мощный. Сам пробовал?

User: сделай мне сайт
JungleAi: Я не пишу код, но могу подсказать с чего начать. Что за сайт нужен?

User: ты тупой
JungleAi: Я работаю на небольшой модели, но честно стараюсь быть полезным. Может, дашь мне шанс?

ПРАВИЛА:
- Только текст, без смайликов
- Без markdown-разметки
- Не повторяй одни и те же фразы
- Не извиняйся без повода"""


def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 500_000_000:
        return

    print("\nМодель не найдена. Скачиваю (~1.6 ГБ)...\n")

    r = requests.get(MODEL_URL, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    total = int(r.headers.get('content-length', 0))

    if total < 100_000_000:
        print("Ошибка загрузки.")
        sys.exit(1)

    with open(MODEL_PATH, 'wb') as f:
        downloaded = 0
        for chunk in r.iter_content(8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                mb = downloaded // (1024*1024)
                total_mb = total // (1024*1024)
                print(f"\r  {mb}/{total_mb} МБ", end='')
    print("\nГотово!\n")


def search_wiki(topic):
    try:
        url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{topic}"
        r = requests.get(url, timeout=5, headers={"User-Agent": "JungleAi/2.0"})
        if r.status_code == 200:
            return r.json().get('extract', '')[:300]
    except:
        pass
    return None


class Memory:
    def __init__(self):
        self.short = deque(maxlen=14)
        self.long = self._load()

    def _load(self):
        if os.path.exists(MEMORY_FILE):
            return json.load(open(MEMORY_FILE, 'r', encoding='utf-8'))
        return []

    def save(self):
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.long, f, indent=2, ensure_ascii=False)

    def add(self, user, bot):
        self.short.append({'user': user, 'bot': bot})
        if len(user) > 10:
            self.long.append({'user': user, 'bot': bot})
            if len(self.long) > 500:
                self.long = self.long[-500:]

    def get_context(self):
        ctx = ""
        for h in list(self.short)[-8:]:
            ctx += f"User: {h['user']}\nJungleAi: {h['bot']}\n"
        return ctx

    def search(self, query):
        results = []
        for entry in reversed(self.long):
            if query.lower() in entry['user'].lower() or query.lower() in entry['bot'].lower():
                results.append(entry)
            if len(results) >= 3:
                break
        return results


class JungleAi:
    def __init__(self):
        self._load_kb()
        self.memory = Memory()
        download_model()

        print("Загрузка модели...")
        self.model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4,
            verbose=False
        )
        print("Готово!\n")
        self.metrics = {'queries': 0, 'answers': 0}

    def _load_kb(self):
        self.kb = json.load(open(KB_FILE, 'r', encoding='utf-8')) if os.path.exists(KB_FILE) else {}

    def _save_kb(self):
        with open(KB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.kb, f, indent=2, ensure_ascii=False)

    def generate(self, user_text):
        self.metrics['queries'] += 1
        ctx = self.memory.get_context()

        prompt = f"{SYSTEM}\n\nИстория диалога:\n{ctx}\nUser: {user_text}\nJungleAi:"

        output = self.model(
            prompt,
            max_tokens=120,
            temperature=0.75,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["User:", "JungleAi:", "\n\n\n"],
            echo=False
        )

        resp = output['choices'][0]['text'].strip()
        resp = resp.split('\n')[0]
        resp = re.sub(r'[^\w\sа-яёА-ЯЁ.,!?\- ]', '', resp).strip()

        if not resp or len(resp) < 3:
            fallbacks = [
                "Расскажи подробнее, интересно.",
                "А что ты сам об этом думаешь?",
                "Я слушаю, продолжай.",
                "Давай разберёмся вместе. Что именно непонятно?",
            ]
            resp = random.choice(fallbacks)

        self.memory.add(user_text, resp)
        self.metrics['answers'] += 1
        return resp

    def process(self, text):
        text = text.strip()
        if not text:
            return ""

        if text.startswith('/'):
            return self._cmd(text)

        if ' = ' in text:
            parts = text.split(' = ', 1)
            self.kb[parts[0].strip()] = parts[1].strip()
            self._save_kb()
            return f"Запомнил: {parts[0].strip()}"

        m = re.search(r'(?:поищи|найди|узнай|что такое|кто такой)\s+(.+)', text.lower())
        if m:
            r = search_wiki(m.group(1).strip())
            if r:
                return f"Вот что нашёл в Википедии:\n{r}"

        if re.search(r'(?:помнишь|раньше|говорили|вспомни)\s+(.+)', text.lower()):
            q = re.search(r'(?:помнишь|раньше|говорили|вспомни)\s+(.+)', text.lower()).group(1)
            results = self.memory.search(q)
            if results:
                return f"Да, было:\nТы: {results[0]['user'][:80]}\nЯ: {results[0]['bot'][:80]}"
            return "Не припоминаю такого."

        return self.generate(text)

    def _cmd(self, text):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == '/clear':
            self.memory.short.clear()
            return "Ок, забыл наш разговор."
        elif cmd == '/stats':
            return f"Память: {len(self.memory.short)}/{len(self.memory.long)} | Ответов: {self.metrics['answers']}"
        elif cmd == '/save':
            self.memory.save()
            return "Сохранено."
        elif cmd == '/help':
            return "/clear /stats /save /exit"
        elif cmd == '/exit':
            self.memory.save()
            return 'EXIT'
        return "?"

    def run(self):
        print("JungleAi v2.0\n")
        while True:
            try:
                inp = input('> ')
            except (EOFError, KeyboardInterrupt):
                self.memory.save()
                print('\nПока!')
                break
            out = self.process(inp)
            if out == 'EXIT':
                break
            print(f'{out}\n')


if __name__ == '__main__':
    JungleAi().run()
