import asyncio
import difflib
from collections import defaultdict, deque
import hashlib
import json
import math
import random
import re
import threading
import time
from pathlib import Path
from typing import Literal, Optional

try:
    import numpy as np
except ImportError:  # 意味検索(ruri-v3)なしでも動くようにする
    np = None

import discord
from discord import app_commands
from discord.ext import commands

basho = Path(__file__).parent
setteifile = basho / "settei.json"
kotobafile = basho / "kotoba.json"
modelfile = basho / "model.json"
trainingfile = basho / "train.json"
nuunchanfile = basho / "nuunchan.json"
banpuid = 302050872383242240

tokun = "ここにBotトークンを入れてください"
gachago = "ぬーんガチャ"
koteiritsu = 1 / 5                              
renpo_window = 5.0
renpo_limit = 7
renpo = defaultdict(deque)
recent_human_messages = {}
tokenre = re.compile(r"[一-龥々]+|[ァ-ヴー]{2,}|[A-Za-z0-9]{2,}")
tomeru = {"今日", "明日", "昨日", "今", "何", "私", "自分", "人", "時", "事", "日", "本当", "最近"}

                                                                                       
owarire = re.compile(r"(?:[ぬヌ][ーｰ〜~ぅぉおぁっゥォオァッ]*[んン]|[。！？!?…])[。！？!?…~〜]*")
# ぬーん = 「ぬ・ヌ・N」で始まり「ん・ン・N」で終わる8文字以内のことば(間は仮名・長音・英字のみ。最短一致で1語ずつ数える)。
nuunre = re.compile(r"[ぬヌnNｎＮ][ぁ-んァ-ヶーｰ〜~a-zA-Zａ-ｚＡ-Ｚ]{0,6}?[んンnNｎＮ]")

Kiri = Literal["オン", "オフ"]

# 知識学習の設定(全サーバー共通。件数の上限なし。発言の形式は問わない)
chishiki_tomeru = {
    "です", "ます", "から", "まで", "ので", "のに", "けど", "でも", "だけ", "より", "など", "って", "とは", "では", "には",
    "して", "した", "する", "ない", "ある", "いる", "こと", "もの", "よう", "これ", "それ", "あれ", "どれ", "なに", "なん",
    "だれ", "どこ", "いつ", "どう", "ここ", "そこ", "でした", "ました", "ません", "いい", "うん", "はい", "あの", "その",
    "この", "どの", "だよ", "よね", "かな", "ですか", "ますか", "について", "どんな", "教えて", "ください", "知って", "知ってる",
    "お願い", "なる", "なっ", "いて", "って何", "とは何", "という", "ている", "てる", "でき", "ちゃん", "さん", "くん",
    "何", "誰", "人", "事", "時", "日", "今", "私", "俺", "僕", "自分", "今日", "明日", "昨日", "最近", "本当", "皆",
    "よく", "もう", "まだ", "すぐ", "ため", "ところ", "みたい", "らしい", "だった", "だろう", "でしょ", "じゃ", "たい", "れる",
    "られ", "せる", "ている", "でる", "なら", "ので", "から", "ます", "ませ", "った", "って", "ない", "なく", "より", "ほど",
}
# ひらがなの連なり(「ってどんな」など)から機能語を取り除くための正規表現。長い語を先に試す。
chishiki_kinou = re.compile("|".join(
    re.escape(w) for w in sorted((w for w in chishiki_tomeru if re.fullmatch(r"[ぁ-ん]+", w)), key=len, reverse=True)
))
chishiki_crawling = set()

# 意味検索(ruri-v3)の設定。LLMは使わない。モデルは公開されていて、アカウント登録・APIキーは不要。
# 初回起動時だけ Hugging Face から自動ダウンロードされる(以降はローカルのキャッシュを使う)。
# ローカルに置いたモデルのフォルダを指定してもよい。
vec_model_name = "cl-nagoya/ruri-v3-130m"
vec_seed_min = 0.65     # train.jsonの基礎文(s)を意味で選ぶときの下限類似度
vec_prefix = ""         # ruri-v3の入力接頭辞。発言同士の比較なので無し。モデルカードで確認し、必要ならここを変える(変えるとキャッシュは自動で作り直される)
vec_max_len = 128       # 入力の最大トークン数(短くするほど速い)
vec_fact_min = 0.55     # 知識をヒントに使う類似度(コサイン)の下限。/知識 検索 で実際の値を見て調整する
vec_turn_min = 0.65     # 似た質問への過去の返答をヒントに使う類似度の下限
vec_margin = 0.06       # 1位との差がこの範囲内の候補だけを併用する
vec_max_hints = 3       # ヒントに使う最大件数
reply_min_chars = 28    # 返信がこれより短いときは、近い候補をもう1〜2件つなげて長くする
reply_max_chars = 140   # つなげたあとの最大文字数
vec_dl_retries = 5      # 初回ダウンロードで429などが出たときの試行回数
vec_dl_wait = 30        # 最初の待ち秒数(以降 2倍ずつ増える。上限600秒)
# レート制限が頻発するなら、無料のHugging Faceアカウントでトークンを作り、環境変数 HF_TOKEN に入れると上限が大きく上がる。
# 一度ダウンロードできたら、以降の起動は通信しない(上のキャッシュ優先読み込み)。
def yomu(file, hajime):
    if not file.is_file():
        file.write_text(json.dumps(hajime, ensure_ascii=False, indent=2), "utf-8")
        return hajime
    return json.loads(file.read_text("utf-8"))


settei = yomu(setteifile, {})
kotoba = yomu(kotobafile, {
    "banpu": {"bun": "Bumpを検知したぬーん", "gazou": None},
    "gacha": {
        "例": {"ritsu": 100, "bun": "ぬーんガチャの例ぬーん", "gazou": None},
    },
    "hannou": [
        {"kiwado": ["こんにちは"], "ritsu": 0, "bun": "こんにちはぬーん", "gazou": None}
    ],
})


                    
for _g in settei.values():
    if isinstance(_g, dict) and "hansou" in _g:
        _g.setdefault("hannou", _g.pop("hansou"))
if "hansou" in kotoba:
    kotoba.setdefault("hannou", kotoba.pop("hansou"))




def nuunchanyomu():
    if not nuunchanfile.is_file():
        data = {"users": {}, "ranking": {}}
        nuunchanfile.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return data
    try:
        data = json.loads(nuunchanfile.read_text("utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("users", {})
    data.setdefault("ranking", {})
    for user in data["users"].values():
        if isinstance(user, dict):
            user.pop("kioku", None)
    return data


nuunchan = nuunchanyomu()


def nuunchankaku():
    tmp = nuunchanfile.with_suffix(".tmp")
    tmp.write_text(json.dumps(nuunchan, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(nuunchanfile)


def nuunchanuser(uid):
    uid = str(uid)
    u = nuunchan["users"].setdefault(uid, {})
    u.setdefault("yobikata", "ぬーんちゃん")
    return u


def nuunchanyobikata(uid):
    return nuunchanuser(uid)["yobikata"]


def nuunchanmatch(text, uid):
    name = nuunchanyobikata(uid)
    return name if name and name in text else None


def nuunchanrank(uid, guild_id, amount):
    uid = str(uid)
    guild_id = str(guild_id)
    ranking = nuunchan.setdefault("ranking", {})
    if "global" not in ranking and all(not isinstance(v, dict) for v in ranking.values()):
        ranking = {"global": ranking, "servers": {}}
        nuunchan["ranking"] = ranking
    ranking.setdefault("global", {})
    ranking.setdefault("servers", {})
    ranking["global"][uid] = int(ranking["global"].get(uid, 0)) + int(amount)
    server = ranking["servers"].setdefault(guild_id, {})
    server[uid] = int(server.get(uid, 0)) + int(amount)


def nuunchanrankpos(ranking, uid):
    uid = str(uid)
    rows = sorted(((str(k), int(v)) for k, v in ranking.items()), key=lambda x: (-x[1], x[0]))
    for i, (rid, count) in enumerate(rows, 1):
        if rid == uid:
            return i, count
    return None, 0


async def nuunchanrankname(guild, uid):
    uid = int(uid)
    if guild:
        member = guild.get_member(uid)
        if member:
            return member.display_name
    user = bot.get_user(uid)
    if user:
        return user.display_name
    try:
        user = await bot.fetch_user(uid)
        return user.display_name
    except (discord.NotFound, discord.HTTPException, discord.Forbidden):
        return f"ユーザー{uid}"


async def nuunchanranktext(uid, guild):
    ranking = nuunchan.get("ranking", {})
    if "global" not in ranking and all(not isinstance(v, dict) for v in ranking.values()):
        ranking = {"global": ranking, "servers": {}}
    global_ranking = ranking.get("global", {})
    server_ranking = ranking.get("servers", {}).get(str(guild.id), {}) if guild else {}
    out = []
    for title, data in (("グローバルランキング", global_ranking), ("サーバー内ランキング", server_ranking)):
        out.append(title)
        if not data:
            out.append("まだランキングは始まっていないぬーん")
        else:
            rows = sorted(((str(k), int(v)) for k, v in data.items()), key=lambda x: (-x[1], x[0]))[:10]
            for i, (rid, count) in enumerate(rows, 1):
                out.append(f"{i}位　{await nuunchanrankname(guild, rid)}　{count}ぬーん")
        pos, count = nuunchanrankpos(data, uid)
        if pos is not None:
            out.append(f"あなた　{pos}位　{count}ぬーん")
        else:
            out.append("あなた　ランキング外　0ぬーん")
        out.append("")
    return "\n".join(out).rstrip()


HAIKU = [
    "ぬんぬんぬ、ぬぬぬぬぬぬぬ、ぬーんぬん",
    "ぬーん咲く、今日も元気に、ぬーんする",
    "春のぬーん、夏もぬーんして、秋ぬーん",
    "ぬーんかな、ぬーんじゃないかな、やっぱぬーん",
    "朝ぬーん、昼もぬーんで、夜ぬーん",
]


def nuunchanart(text, uid):
    name = nuunchanmatch(text, uid)
    if not name:
        return None
    body = text.replace(name, "", 1).strip(" 、,，。！？!?　")
    if "俳句" in body:
        return random.choice(HAIKU)
    if body in ("", "呼んだ", "おい", "ねえ", "ねぇ"):
        return "呼んだぬーん？"
    return random.choice([
        "ぬーん！呼んだぬーん？",
        "ぬーんちゃんはここにいるぬーん！",
        "どうしたぬーん？",
    ])

def kaku():
    setteifile.write_text(json.dumps(settei, ensure_ascii=False, indent=2), "utf-8")


def globalchat_channels():
    data = settei.setdefault("globalchat", {})
    channels = data.setdefault("channels", [])
    if not isinstance(channels, list):
        channels = []
        data["channels"] = channels
    return channels


def globalchat_enabled(channel_id):
    return str(channel_id) in {str(x) for x in globalchat_channels()}


def globalchat_add(channel_id):
    channels = globalchat_channels()
    cid = str(channel_id)
    if cid not in {str(x) for x in channels}:
        channels.append(cid)
        kaku()


def globalchat_remove(channel_id):
    channels = globalchat_channels()
    cid = str(channel_id)
    before = len(channels)
    channels[:] = [x for x in channels if str(x) != cid]
    if len(channels) != before:
        kaku()
        return True
    return False


globalchat_links = {}
globalchat_roots = {}
globalchat_message_channels = {}
globalchat_reaction_lock = set()


def globalchat_root_id(message_id):
    return globalchat_roots.get(message_id, message_id)


def globalchat_clean_mentions(text):
    text = str(text or "")
    text = text.replace("@everyone", "＠everyone").replace("@here", "＠here")
    text = re.sub(r"<@&(\d+)>", lambda m: "＠<@&" + m.group(1) + ">", text)
    return text


async def globalchat_get_message(channel, message_id):
    if not channel:
        return None
    try:
        return await channel.fetch_message(message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None


async def globalchat_build_embeds(message, edited=False, destination=None):
    if not message.content and not message.attachments:
        return []
    name = getattr(message.author, "display_name", message.author.name)
    icon = message.author.display_avatar.url
    body = globalchat_clean_mentions(message.content)
    embed = discord.Embed(description=body[:4096] if body else None)
    embed.set_author(name=name, icon_url=icon)
    if edited:
        embed.set_footer(text="編集済み")

    if message.reference and message.reference.message_id:
        reply = message.reference.resolved
        if not isinstance(reply, discord.Message):
            reply = await globalchat_get_message(message.channel, message.reference.message_id)
        reply_link = None
        reply_name = None
        reply_text = None
        if isinstance(reply, discord.Message):
            reply_name = getattr(reply.author, "display_name", reply.author.name)
            reply_text = globalchat_clean_mentions(reply.content or "添付ファイル")
            reply_root = globalchat_root_id(reply.id)
            if reply.id != reply_root and reply.embeds:
                source_embed = reply.embeds[0]
                if source_embed.author and source_embed.author.name:
                    reply_name = source_embed.author.name
                if source_embed.description:
                    reply_text = globalchat_clean_mentions(source_embed.description)
            reply_text = reply_text.replace("\n", " ").replace("\r", " ").strip()
            if len(reply_text) > 300:
                reply_text = reply_text[:297] + "..."
            if destination is not None:
                if reply.id == reply_root and reply.channel.id == destination.id:
                    reply_link = reply.jump_url
                else:
                    for mapped_id in globalchat_links.get(reply_root, []):
                        if globalchat_message_channels.get(mapped_id) == destination.id:
                            mapped_message = await globalchat_get_message(destination, mapped_id)
                            if mapped_message:
                                reply_link = mapped_message.jump_url
                                break
            if reply_link is None and destination is None:
                reply_link = reply.jump_url
        if reply_link and reply_name is not None and reply_text is not None:
            embed.add_field(
                name="返信先",
                value=f"[{globalchat_clean_mentions(reply_name)}]({reply_link})\n{reply_text}",
                inline=False,
            )

    image_attachments = [
        a for a in message.attachments
        if (a.content_type or "").startswith("image/")
    ]
    if not body and not image_attachments:
        embed.description = "添付ファイル"

    embeds = [embed]
    for attachment in image_attachments[:9]:
        image_embed = discord.Embed()
        image_embed.set_image(url=f"attachment://{attachment.filename}")
        embeds.append(image_embed)
    return embeds


async def globalchat_send(message, destination):
    embeds = await globalchat_build_embeds(message, destination=destination)
    if not embeds:
        return None
    files = []
    for attachment in message.attachments:
        try:
            files.append(await attachment.to_file(use_cached=True))
        except Exception:
            continue
    try:
        sent = await destination.send(
            embeds=embeds,
            files=files,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=False),
        )
    except (discord.HTTPException, discord.Forbidden):
        return None
    globalchat_copy_ids.append(sent.id)
    return sent


globalchat_copy_ids = deque(maxlen=5000)


def globalchat_is_copy(message):
    """Bot自身が中継したコピーか。コピーを再び中継すると、全チャンネルで無限に増え続ける。
    中継の登録は全チャンネルへ送り終えた後なので、登録だけでは間に合わない。
    そのため、中継用の埋め込み(rich・作者名つき)を持つBot発言も構造で判定する。"""
    if message.id in globalchat_copy_ids:
        return True
    if globalchat_roots.get(message.id, message.id) != message.id:
        return True
    if message.author.bot:
        for e in message.embeds:
            if getattr(e, "type", None) == "rich" and e.author and e.author.name:
                return True
    return False


async def globalchat_relay(message):
    if not message.guild or not globalchat_enabled(message.channel.id):
        return
    if message.author.bot and message.author.id != bot.user.id:
        return
    if globalchat_is_copy(message):
        return
    if message.id in globalchat_links:
        return
    targets = []
    for cid in globalchat_channels():
        channel = bot.get_channel(int(cid))
        if isinstance(channel, (discord.TextChannel, discord.Thread)) and channel.id != message.channel.id:
            targets.append(channel)
    if not targets:
        return
    sent_messages = []
    for channel in targets:
        sent = await globalchat_send(message, channel)
        if sent:
            sent_messages.append(sent)
    if sent_messages:
        ids = [message.id] + [m.id for m in sent_messages]
        globalchat_roots[message.id] = message.id
        globalchat_message_channels[message.id] = message.channel.id
        for mid in ids:
            globalchat_roots[mid] = message.id
            globalchat_links[mid] = ids
        for sent in sent_messages:
            globalchat_message_channels[sent.id] = sent.channel.id


async def globalchat_reaction(payload, add=True):
    if not bot.user or payload.user_id == bot.user.id:
        return
    ids = globalchat_links.get(payload.message_id)
    if not ids or payload.message_id in globalchat_reaction_lock:
        return
    key = (payload.message_id, payload.user_id, str(payload.emoji), add)
    if key in globalchat_reaction_lock:
        return
    globalchat_reaction_lock.add(key)
    try:
        for mid in ids:
            if mid == payload.message_id:
                continue
            root = globalchat_root_id(mid)
            if root != globalchat_root_id(payload.message_id):
                continue
            target = None
            for cid in globalchat_channels():
                ch = bot.get_channel(int(cid))
                if not isinstance(ch, (discord.TextChannel, discord.Thread)):
                    continue
                target = await globalchat_get_message(ch, mid)
                if target:
                    break
            if not target:
                continue
            try:
                if add:
                    await target.add_reaction(payload.emoji)
                else:
                    await target.remove_reaction(payload.emoji, bot.user)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass
    finally:
        globalchat_reaction_lock.discard(key)


async def globalchat_edit(message):
    root = globalchat_root_id(message.id)
    ids = globalchat_links.get(root)
    if not ids or message.id != root:
        return
    if not message.guild:
        return
    embeds_by_channel = {}
    if not globalchat_links.get(root):
        return
    files = []
    for attachment in message.attachments:
        try:
            files.append(await attachment.to_file(use_cached=True))
        except Exception:
            continue
    for mid in ids:
        if mid == root:
            continue
        target = None
        for cid in globalchat_channels():
            ch = bot.get_channel(int(cid))
            if not isinstance(ch, (discord.TextChannel, discord.Thread)):
                continue
            target = await globalchat_get_message(ch, mid)
            if target:
                break
        if not target:
            continue
        try:
            embeds = await globalchat_build_embeds(message, edited=True, destination=target.channel)
            if not embeds:
                continue
            await target.edit(embeds=embeds, attachments=files, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=False))
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass


async def globalchat_delete(message):
    root = globalchat_root_id(message.id)
    ids = globalchat_links.get(root)
    if not ids or message.id != root:
        return
    for mid in ids:
        if mid == root:
            continue
        for cid in globalchat_channels():
            ch = bot.get_channel(int(cid))
            if not isinstance(ch, (discord.TextChannel, discord.Thread)):
                continue
            target = await globalchat_get_message(ch, mid)
            if not target:
                continue
            try:
                await target.delete()
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass
    for mid in ids:
        globalchat_links.pop(mid, None)
        globalchat_roots.pop(mid, None)
        globalchat_message_channels.pop(mid, None)

def saba(x):
    return settei.setdefault(str(x.guild.id), {"banpu": True, "hannou": {}})


def han(x):
    return saba(x)["hannou"].get(str(x.channel.id), True)


def onofu(b):
    return "オン" if b else "オフ"


def ritsu(x):
    """このチャンネルに設定された返信確率(0〜100)。未設定ならNone。"""
    v = saba(x).get("ritsu", {}).get(str(x.channel.id))
    try:
        return None if v is None else max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return None


def pasento(v):
    try:
        return None if v is None else max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return None


def ritsusaba(x):
    """サーバー全体の返信確率(0〜100)。未設定ならNone。"""
    return pasento(saba(x).get("ritsu_saba"))


def ritsukojin(x):
    """発言者個人の返信確率(0〜100)。未設定ならNone。interactionでもmessageでも使える。"""
    hito = getattr(x, "user", None) or x.author
    return pasento(saba(x).get("ritsu_kojin", {}).get(str(hito.id)))


def ritsukimaru(x):
    """実際に使う返信確率と、その出どころ。優先順位: 個人 > チャンネル > サーバー > 初期値。"""
    for moto, v in (("個人", ritsukojin(x)), ("チャンネル", ritsu(x)), ("サーバー", ritsusaba(x))):
        if v is not None:
            return v, moto
    try:
        return round(float(nuun.settings.get("generation_chance", 0.22)) * 100), "初期値"
    except (TypeError, ValueError):
        return 22, "初期値"


def ritsumoji(x):
    """今実際に使われる返信確率(パーセント)。"""
    return ritsukimaru(x)[0]


def ritsusetsu(x):
    v, moto = ritsukimaru(x)
    return f"返信確率は{v}パーセント({moto})"


def dashi(r):
    k = {}
    if r.get("bun"):
        k["content"] = r["bun"]
    path = basho / r["gazou"] if r.get("gazou") else None
    if path and path.is_file():
        k["file"] = discord.File(path)
    return k


async def okuru(heya, r):
    k = dashi(r)
    if k:
        await heya.send(**k)


def hiku():
    if not kotoba.get("gacha"):
        return {"content": "ガチャが設定されていないぬーん"}
    entries = [(name, item) for name, item in kotoba["gacha"].items() if isinstance(item, dict)]
    if not entries:
        return {"content": "ガチャの設定が正しくないぬーん"}
    names = [name for name, _ in entries]
    weights = []
    for _, item in entries:
        try:
            weights.append(max(0.0, float(item.get("ritsu", 1))))
        except (TypeError, ValueError):
            weights.append(0.0)
    if not any(weights):
        weights = [1.0] * len(entries)
    na = random.choices(names, weights=weights, k=1)[0]
    return dashi(dict(kotoba["gacha"][na])) or {"content": f"{na}ぬーん"}


class SharedMemory:
    """全サーバー共通の文章・関連語・会話の流れを保存する。"""
    def __init__(self):
        self.path = basho / "memory.json"
        self.sentences = []
        self.counts = {}
        self.relations = {}
        self.turns = []
        self.facts = []
        self.dirty = 0
        self.save_lock = threading.Lock()
        self.save_state = threading.Lock()
        self.save_busy = False
        self.save_again = False
        try:
            data = json.loads(self.path.read_text("utf-8")) if self.path.is_file() else {}
            if isinstance(data, dict):
                self.sentences = data.get("sentences", [])
                self.counts = data.get("counts", {})
                self.relations = data.get("relations", {})
                self.turns = data.get("turns", [])
                self.facts = data.get("facts", [])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        if not isinstance(self.sentences, list):
            self.sentences = []
        if not isinstance(self.counts, dict):
            self.counts = {}
        if not isinstance(self.relations, dict):
            self.relations = {}
        if not isinstance(self.turns, list):
            self.turns = []
        if not isinstance(self.facts, list):
            self.facts = []
        self.facts = [
            {
                "s": x["s"] if isinstance(x.get("s"), str) else "",
                "t": x["t"],
                "g": x["g"] if isinstance(x.get("g"), int) else 0,
                "by": [u for u in x.get("by", []) if isinstance(u, int)][-5:],
                "x": 1 if x.get("x") else 0,
            }
            for x in self.facts
            if isinstance(x, dict) and isinstance(x.get("t"), str) and x["t"].strip()
        ]
        self.sentences = [str(x) for x in self.sentences if isinstance(x, str)]
        self.turns = [x for x in self.turns if isinstance(x, dict) and isinstance(x.get("q"), str) and isinstance(x.get("a"), str)]
        self.chishiki_kouchiku()

    @staticmethod
    def normalize(text):
        text = str(text or "").casefold()
        text = re.sub(r"<@!?\d+>|<#\d+>|<@&\d+>", "", text)
        text = re.sub(r"[\s　、。,.，．!！?？…~〜:：;；]+", "", text)
        return text.strip("『』「」\"'()（）")

    @staticmethod
    def terms(text):
        text = SharedMemory.normalize(text)
        found = re.findall(r"[一-龥々〆ヵヶ]+|[ぁ-ん]+|[ァ-ヴー]+|[a-z0-9]+", text)
        terms = []
        for term in found:
            if len(term) >= 2 and term not in terms:
                terms.append(term)
            if len(term) >= 4:
                for i in range(len(term) - 1):
                    part = term[i:i + 2]
                    if part not in terms:
                        terms.append(part)
        for i in range(len(text) - 1):
            part = text[i:i + 2]
            if part not in terms and not part.isspace():
                terms.append(part)
        return terms[:30]

    @staticmethod
    def grams(text):
        text = SharedMemory.normalize(text)
        if len(text) < 2:
            return {text} if text else set()
        return {text[i:i + 2] for i in range(len(text) - 1)}

    def learn(self, text):
        cleaned = str(text or "").strip()
        normalized = self.normalize(cleaned)
        if len(normalized) < 2:
            return
        if cleaned in self.counts:
            self.counts[cleaned] = int(self.counts.get(cleaned, 0)) + 1
        else:
            self.counts[cleaned] = 1
            self.sentences.append(cleaned)
        terms = self.terms(cleaned)
        for i, left in enumerate(terms):
            rel = self.relations.setdefault(left, {})
            for right in terms[i + 1:]:
                if left != right:
                    rel[right] = min(100000, int(rel.get(right, 0)) + 1)
                    other = self.relations.setdefault(right, {})
                    other[left] = min(100000, int(other.get(left, 0)) + 1)
        if len(self.sentences) > 12000:
            self.sentences = self.sentences[-12000:]
            keep = set(self.sentences)
            self.counts = {k: v for k, v in self.counts.items() if k in keep}
        if len(self.relations) > 20000:
            self.relations = dict(sorted(self.relations.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:16000])
        self.dirty += 1
        if self.dirty >= 20:
            self.save_async()

    def learn_turn(self, question, answer):
        q = str(question or "").strip()
        a = str(answer or "").strip()
        if len(self.normalize(q)) < 2 or len(self.normalize(a)) < 2:
            return
        qn, an = self.normalize(q), self.normalize(a)
        if qn == an:
            return
        for item in self.turns:
            if self.normalize(item["q"]) == qn and self.normalize(item["a"]) == an:
                item["n"] = min(100000, int(item.get("n", 1)) + 1)
                self.dirty += 1
                return
        self.turns.append({"q": q, "a": a, "n": 1})
        vec_index.add("t", q, a)
        if len(self.turns) > 8000:
            self.turns = self.turns[-8000:]
        self.learn(q)
        self.learn(a)
        self.dirty += 1

    def similarity(self, query, candidate):
        qn, cn = self.normalize(query), self.normalize(candidate)
        if not qn or not cn:
            return 0.0
        if qn == cn:
            return 1.0
        if cn.startswith(qn):
            return 0.92 + min(0.07, (len(cn) - len(qn)) / 1000)
        if qn in cn:
            return 0.82
        qt, ct = set(self.terms(query)), set(self.terms(candidate))
        overlap = len(qt & ct) / max(1, len(qt | ct))
        qg, cg = self.grams(query), self.grams(candidate)
        gram = len(qg & cg) / max(1, len(qg | cg))
        related = 0.0
        if qt and ct:
            best = 0.0
            for token in qt:
                row = self.relations.get(token, {})
                total = sum(int(v) for v in row.values())
                if not total:
                    continue
                for other in ct:
                    count = int(row.get(other, 0))
                    if count:
                        best = max(best, min(1.0, count / max(3, total * 0.25)))
            related = best
        return overlap * 0.45 + gram * 0.45 + related * 0.10

    def query_variants(self, question):
        normalized = self.normalize(question)
        variants = [normalized]
        endings = (
            r"ってどんな(?:感じ|もの|人|やつ)?$", r"とはどんな(?:感じ|もの|人|やつ)?$",
            r"って何(?:者|ですか|なの|なのか)?$", r"とは何(?:者|ですか|なの|なのか)?$",
            r"ってなに(?:者|ですか|なの)?$", r"とはなに(?:者|ですか|なの)?$",
            r"はどんな(?:感じ|もの|人|やつ)?$", r"は誰(?:ですか|なの)?$",
            r"は何(?:者|ですか|なの|なのか)?$", r"について教えて$",
            r"について$", r"とは$", r"って$", r"ってどういうこと$",
            r"どんな感じ$", r"教えて$", r"知ってる$", r"なに$", r"何$", r"誰$",
        )
        for pattern in endings:
            shortened = re.sub(pattern, "", normalized)
            if len(shortened) >= 2 and shortened not in variants:
                variants.append(shortened)
        return variants

    def query_similarity(self, question, candidate):
        return max(self.similarity(q, candidate) for q in self.query_variants(question))

    def recall(self, question, context=None):
        query = str(question or "").strip()
        qn = self.normalize(query)
        variants = self.query_variants(query)
        if len(qn) < 2:
            return None

        # 先に学習済みの会話の流れを探し、質問と似た発言に続いた返答を優先する。
        turn_candidates = []
        for item in self.turns[-1500:]:
            question_score = self.query_similarity(query, item["q"])
            if question_score < 0.58:
                continue
            context_score = 0.0
            if context:
                context_score = max((self.query_similarity(context, item[k]) for k in ("q", "a")), default=0.0)
            score = question_score + min(0.06, context_score * 0.06)
            score += min(0.025, max(0, int(item.get("n", 1)) - 1) * 0.002)
            turn_candidates.append((score, item["a"], item["q"]))
        if turn_candidates:
            score, answer, source = max(turn_candidates, key=lambda x: x[0])
            if score >= 0.64 and self.normalize(answer) not in {self.normalize(query), "ぬーんをやりるぬーん"}:
                return answer

        # 次に既出の文章を、先頭一致だけでなく単語・文字列・関連語の重なりで検索する。
        candidates = []
        for sentence in self.sentences[-4000:]:
            normalized = self.normalize(sentence)
            if len(normalized) <= len(qn) or normalized == qn:
                continue
            matching_prefix = next((v for v in variants if normalized.startswith(v) and len(v) >= 2), None)
            query_score = self.query_similarity(query, sentence)
            if matching_prefix:
                score = 0.78 + min(0.12, (len(normalized) - len(matching_prefix)) / 200)
            else:
                if query_score < 0.68:
                    continue
                score = query_score
                if context:
                    score += min(0.04, self.query_similarity(context, sentence) * 0.04)
            if score >= 0.68:
                candidates.append((score + min(0.02, self.counts.get(sentence, 1) * 0.001), sentence))
        if not candidates:
            return None
        score, sentence = max(candidates, key=lambda item: item[0])
        if score < 0.70:
            return None
        normalized_sentence = self.normalize(sentence)
        matching_variant = next((v for v in variants if normalized_sentence.startswith(v) and len(v) >= 2), None)
        if matching_variant:
            prefix_length = None
            for i in range(1, len(sentence) + 1):
                if self.normalize(sentence[:i]) == matching_variant:
                    prefix_length = i
                    break
            if prefix_length is not None:
                answer = sentence[prefix_length:].strip(" 　、。,.，．!！?？…~〜:：;；")
                if answer:
                    return answer
        return sentence

    def retrieve_evidence(self, question, context=None, limit=5):
        """回答を決めず、質問に関連する過去の発言を根拠候補として返す。"""
        query = str(question or "").strip()
        query_norm = self.normalize(query)
        if len(query_norm) < 2:
            return []

        found = []
        seen = set()

        # 会話ペアは「質問文」と「その後の返答」を別々に評価する。
        for item in self.turns[-3000:]:
            source_question = str(item.get("q", "")).strip()
            qscore = self.query_similarity(query, source_question)
            cscore = self.query_similarity(context, source_question) if context else 0.0
            # 直近の文脈が似ているだけでは採用しない。今回の質問自体が十分に近い必要がある。
            score = qscore * 0.90 + cscore * 0.04
            answer = str(item.get("a", "")).strip()
            key = self.normalize(answer)
            if answer and key not in seen and qscore >= 0.52 and score >= 0.50:
                found.append({"score": score, "text": answer, "source": source_question, "kind": "turn"})
                seen.add(key)

        # 単独発言も候補にする。単語の重なりだけでなく、文字2-gramの重なりも使う。
        for sentence in self.sentences[-6000:]:
            sentence = str(sentence).strip()
            key = self.normalize(sentence)
            if not key or key == query_norm or key in seen or len(key) < 3:
                continue
            score = self.query_similarity(query, sentence)
            if context:
                score = score * 0.96 + self.query_similarity(context, sentence) * 0.04
            # 単語が少し重なるだけの無関係な発言を、回答根拠として拾わない。
            if score >= 0.48:
                found.append({"score": score, "text": sentence, "source": "", "kind": "sentence"})
                seen.add(key)

        # 高得点だけを採用し、同じ話の言い換えが候補を埋めないようにする。
        found.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        selected_grams = []
        for item in found:
            grams = self.grams(item["text"])
            if any(len(grams & old) / max(1, len(grams | old)) > 0.78 for old in selected_grams):
                continue
            selected.append(item)
            selected_grams.append(grams)
            if len(selected) >= max(1, min(int(limit), 5)):
                break
        return selected

    # ---- 知識の学習(全サーバー共通・発言の形式は問わない) ------------------------
    # 質問の「聞く部分」(何・誰・どんな・教えて・知ってる…)と、話題を示す助詞。語尾を1つずつ決め打ちせず、尾から順にはがして話題を取り出す。
    chishiki_gimon = (
        r"(?:何者|なにもの|何|なに|なん|誰|だれ|どこ|いつ|どんな(?:もの|人|感じ|やつ|の)?|どういう(?:もの|人|こと|意味|やつ)?|どう|意味|"
        r"について(?:教えて(?:くれ|ください|ほしい)?)?|教えて(?:くれ|ください|ほしい)?|知ってる|しってる|知ってますか|わかる|分かる|"
        r"ですか|ますか|でしょうか|のか|なのか|かな|だっけ|っけ)"
    )
    chishiki_gimon_end = r"(?:なの|なん|なんだ|だ|です|の|ん|よ|ね|さ|ぞ|や|か)*"
    chishiki_joshi_re = r"(?:というのは|っていうのは|ってのは|とは|って|のこと|こと|について|は|が|を|も|の)+"

    @staticmethod
    def chishiki_wadai(text):
        """質問から話題を取り出す。戻り値: (話題, 聞く形だったか)。例: 「ふにゃって何なの」→(ふにゃ, True)、「ふにゃは？」→(ふにゃ, False)"""
        t = SharedMemory.normalize(text)
        t2 = re.sub(SharedMemory.chishiki_gimon + SharedMemory.chishiki_gimon_end + "$", "", t)
        asked = t2 != t
        t3 = re.sub(SharedMemory.chishiki_joshi_re + "$", "", t2)
        return (t3 if len(t3) >= 2 else t2), asked

    @staticmethod
    def chishiki_shudai(text):
        """教えられた文から (話題, 残り) を取り出す。「AはBだ」「AってBなんだよ」「Aというのは、B」「A=B」など形を問わない。無ければ ("", 文全体)。"""
        t = str(text or "").strip()
        m = re.match(r"^(.{1,20}?)(?:っていうのは|というのは|ってのは|とは|って|は|が|＝|=|：|:)[\s　、,，]*(.+)$", t)
        if not m:
            return "", t
        subject = SharedMemory.normalize(m.group(1))
        if len(subject) < 2 or subject in chishiki_tomeru:
            return "", t
        return subject, m.group(2).strip()

    @staticmethod
    def chishiki_shitsumon(text):
        t = str(text or "").strip()
        if re.search(r"[?？]", t):
            return True
        return SharedMemory.chishiki_wadai(t)[1]

    @staticmethod
    def chishiki_goi(text):
        """文から内容語を取り出す(漢字・カタカナ・英数・ひらがな2文字以上。1文字の漢字もOK)。"""
        t = str(text or "").casefold()
        t = re.sub(r"<@!?\d+>|<#\d+>|<@&\d+>|https?://\S+", " ", t)
        t = re.sub(r"[wｗ]{2,}", " ", t)
        out = []

        def add(x):
            if x not in out:
                out.append(x)

        for run in re.findall(r"[一-龥々〆ヵヶ]+|[ぁ-ん][ぁ-んー]*|[ァ-ヴ][ァ-ヴー]*|[a-z0-9]+", t):
            if run in chishiki_tomeru:
                continue
            if re.fullmatch(r"[ぁ-ん][ぁ-んー]*", run):
                for piece in chishiki_kinou.sub(" ", run).split():
                    if len(piece) >= 3:
                        piece = re.sub(r"[はがをにでともへやの]$", "", piece)
                    if len(piece) >= 2 and piece not in chishiki_tomeru:
                        add(piece)
                continue
            add(run)
            if len(run) >= 3 and re.fullmatch(r"[一-龥々〆ヵヶ]+", run):
                for i in range(len(run) - 1):
                    add(run[i:i + 2])
        return out[:60]

    def chishiki_kouchiku(self):
        """検索用の索引(語→知識)を作り直す。"""
        self.chishiki_idx = {}
        self.chishiki_keys = {}
        self.chishiki_terms = {}
        self.chishiki_shu = {}
        for f in self.facts:
            self._chishiki_toroku(f)

    def _chishiki_toroku(self, f):
        terms = set(self.chishiki_goi(f["t"]))
        if f.get("s"):
            terms.update(self.chishiki_goi(f["s"]))
        self.chishiki_terms[id(f)] = frozenset(terms)
        derived, rest = self.chishiki_shudai(f["t"])
        self.chishiki_shu[id(f)] = (self.normalize(f.get("s", "")), derived, rest)
        self.chishiki_keys[self.normalize(f["t"])] = f
        for t in terms:
            self.chishiki_idx.setdefault(t, []).append(f)

    def chishiki_oboeru(self, text, uid, gid, subject="", explicit=False):
        """知識を1件保存する。戻り値: "新規" / "更新" / None(保存しなかった)。"""
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        subject = str(subject or "").strip()
        key = self.normalize(text)
        if not key:
            return None
        f = self.chishiki_keys.get(key)
        if f is not None:
            if uid not in f["by"]:
                f["by"] = (f["by"] + [uid])[-5:]
            if explicit:
                f["x"] = 1
                if subject and not f.get("s"):
                    f["s"] = subject
                    self.chishiki_kouchiku()
            self.dirty += 1
            return "更新"
        f = {"s": subject, "t": text, "g": gid, "by": [uid], "x": 1 if explicit else 0}
        self.facts.append(f)
        self._chishiki_toroku(f)
        vec_index.add("f", text)
        self.dirty += 1
        if self.dirty >= 20:
            self.save_async()
        return "新規"

    def chishiki_manabu(self, text, uid, gid):
        """発言を文ごとに知識として保存する(形式は問わない)。(新規件数, 更新件数)を返す。"""
        new = upd = 0
        for line in re.split(r"[\n\r]+|(?<=[。！!])", str(text or "")):
            line = line.strip()
            if len(line) < 4 or len(line) > 200:
                continue
            if re.search(r"https?://|<@|<#|@everyone|@here", line) or self.chishiki_shitsumon(line):
                continue
            if re.match(r"(?:私|わたし|俺|おれ|オレ|僕|ぼく|ボク|あたし|うち|自分|君|きみ|お前|おまえ|あなた)", line):
                continue
            rest = nuunre.sub("", line)
            if len(self.normalize(rest)) < 5 or len(self.chishiki_goi(rest)) < 1:    # ひらがなだけの文は内容語が1つにしかならないので、1つあれば覚える
                continue
            r = self.chishiki_oboeru(line, uid, gid)
            if r == "新規":
                new += 1
            elif r == "更新":
                upd += 1
        return new, upd

    def chishiki_sagasu(self, question, limit=2):
        """質問の内容語がよく重なる知識を探す。重なりが弱いときは何も返さない。"""
        qt = self.chishiki_goi(question)
        total = len(self.facts)
        if not qt or not total:
            return []

        def idf(t):
            return math.log((total + 1) / (len(self.chishiki_idx.get(t, ())) + 1)) + 1

        weight = {t: idf(t) for t in qt}
        denom = sum(weight.values())
        known = sorted((t for t in qt if t in self.chishiki_idx), key=lambda t: weight[t], reverse=True)[:4]
        qkey = self.normalize(question)
        candidates = {}
        for t in known:
            for f in self.chishiki_idx[t][-2000:]:
                candidates[id(f)] = f
        scored = []
        for f in candidates.values():
            if self.normalize(f["t"]) == qkey:
                continue
            shared = self.chishiki_terms[id(f)] & weight.keys()
            sw = sum(weight[t] for t in shared)
            coverage = sw / denom if denom else 0.0
            explicit = bool(f["x"])
            if not ((coverage >= 0.7 and sw >= 3.0) or (explicit and coverage >= 0.5 and sw >= 1.0)):
                continue
            scored.append((coverage + (0.15 if explicit else 0.0) + 0.03 * min(len(f["by"]), 5), f))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top = scored[0]
        picked = [top]
        for score, f in scored[1:]:
            if len(picked) >= max(1, limit):
                break
            a, b = self.chishiki_terms[id(top)], self.chishiki_terms[id(f)]
            if score >= top_score * 0.85 and len(a & b) / max(1, len(a | b)) < 0.8:
                picked.append(f)
        return picked

    def chishiki_zenpou(self, question, limit=3, force=False):
        """質問から話題を取り出し、その話題について教えられた知識を探す(類似度に頼らない確実な一致)。
        質問の形も教えた文の形も固定しない。「ふにゃは？」「ふにゃって何」「ふにゃってなんなの」「ふにゃについて教えて」…→「ふにゃはすらいむ」「ふにゃってすらいむのこと」「ふにゃ=すらいむ」…
        force=True なら、聞く言い回しが無くても(「ふにゃは？」のような)話題として扱う。戻り値: [(知識の全文, 答えの部分)]"""
        topic, asked = self.chishiki_wadai(question)
        if not (asked or force) or len(topic) < 2 or topic in chishiki_tomeru:
            return []
        qkey = self.normalize(question)
        loose_ok = len(topic) >= 3 or not re.fullmatch(r"[ぁ-ん]+", topic)
        found = []
        for key, f in self.chishiki_keys.items():
            if key == qkey or len(key) <= len(topic):
                continue
            subj, derived, rest = self.chishiki_shu[id(f)]
            if subj == topic or derived == topic:
                rank, answer = 3, (rest if derived == topic and subj in ("", topic) else f["t"])
            elif key.startswith(topic):
                rank, answer = 2, f["t"]
            elif loose_ok and topic in key:
                rank, answer = 1, f["t"]
            else:
                continue
            if len(self.normalize(answer)) < 2:
                answer = f["t"]
            found.append((rank, 1 if f.get("x") else 0, len(f["by"]), f["t"], answer.strip(" 　、。,.，．!！?？…~〜:：;；")))
        if not found:
            return []
        found.sort(key=lambda x: x[:3], reverse=True)
        best = found[0][0]
        return [(t, a) for r, _, _, t, a in found if r == best][:max(1, limit)]

    def chishiki_kensaku(self, keyword):
        kn = self.normalize(keyword)
        if not kn:
            return []
        return [f for f in self.facts if kn in self.normalize(f["t"]) or kn == self.normalize(f.get("s", ""))]

    def chishiki_wasureru(self, keyword, uid, gid, admin=False):
        """キーワードを含む知識を消す。管理者は自分のサーバーで覚えさせたもの、一般は自分1人だけが教えたもののみ。"""
        targets = [f for f in self.chishiki_kensaku(keyword) if (admin and f["g"] == gid) or f["by"] == [uid]]
        if not targets:
            return 0
        ids = {id(f) for f in targets}
        self.facts = [f for f in self.facts if id(f) not in ids]
        self.chishiki_kouchiku()
        vec_index.discard({("f", f["t"], "") for f in targets})
        self.dirty += 1
        return len(targets)

    def save(self):
        """memory.json を書く。一度に全部をJSON化せず、小分けにして書くので、他の処理(返信)を長く止めない。"""
        with self.save_lock:
            self.dirty = 0
            dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
            sentences, counts = list(self.sentences), dict(self.counts)
            turns, facts, rel = list(self.turns), list(self.facts), list(self.relations.items())
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write('{"sentences":' + dump(sentences) + ',"counts":' + dump(counts) + ',"turns":[')
                f.write(",".join(dump(x) for x in turns))
                f.write('],"facts":[')
                f.write(",".join(dump(x) for x in facts))
                f.write('],"relations":{')
                for i in range(0, len(rel), 300):
                    if i:
                        f.write(",")
                    f.write(",".join(dump(k) + ":" + dump(v) for k, v in rel[i:i + 300]))
                f.write("}}")
            tmp.replace(self.path)

    def save_async(self):
        """裏のスレッドで保存する。保存中にまた学習したら、終わったあとにもう一度だけ保存する。"""
        with self.save_state:
            if self.save_busy:
                self.save_again = True
                return
            self.save_busy = True

        def run():
            while True:
                with self.save_state:
                    self.save_again = False
                try:
                    self.save()
                except OSError as error:
                    print(f"memory.json を保存できません: {error}")
                with self.save_state:
                    if not self.save_again:
                        self.save_busy = False
                        return

        threading.Thread(target=run, daemon=True).start()


shared_memory = SharedMemory()


class VecIndex:
    """ruri(静的埋め込み)による意味検索。知識(f)と会話(t: 質問→返答)をベクトルにして保持する。
    LLMは使わない。意味検索モデルが入っていない・モデルを読めないときは ready=False になり、語の重なりでの検索に切り替わる。"""

    def __init__(self):
        self.path = basho / "memory_vec.npz"
        self.model = None
        self.state = "未読込"
        self.items = []
        self.mat = None
        self.keys = set()
        self.pending_items = []
        self.pending_vecs = []
        self.train_mtime = None
        self.todo = []              # まだベクトルにしていない追加分(裏でまとめてエンコードする)
        self.qcache = {}            # 直近の検索文のベクトル(同じ文の再計算を避ける)
        self.dirty = 0
        self.lock = threading.Lock()

    @property
    def ready(self):
        return self.model is not None and self.state == "使用中"

    @staticmethod
    def key_of(item):
        return (item["k"], item["t"], item.get("a", ""))

    def encode_query(self, text):
        v = self.qcache.get(text)
        if v is None:
            v = self.encode([text])[0]
            if len(self.qcache) >= 256:
                self.qcache.pop(next(iter(self.qcache)))
            self.qcache[text] = v
        return v

    def encode(self, texts):
        vecs = np.asarray(self.model.encode([vec_prefix + t for t in texts], batch_size=32, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def _flush(self):
        if not self.pending_items:
            return
        block = np.vstack(self.pending_vecs).astype(np.float32)
        self.mat = block if self.mat is None else np.vstack([self.mat, block])
        self.items.extend(self.pending_items)
        self.pending_items, self.pending_vecs = [], []

    def load(self):
        """モデルとキャッシュを読む(重いので別スレッドで呼ぶ)。成功したら True。"""
        if np is None:
            self.state = "使えない(numpyが入っていません)"
            return False
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.state = "使えない(pip install sentence-transformers が必要)"
            return False
        self.state = "読込中"
        try:
            self.model = self._load_model(SentenceTransformer)
            self.model.max_seq_length = vec_max_len
        except Exception as error:
            self.model = None
            self.state = f"使えない(モデルを読み込めません: {type(error).__name__})"
            print(f"意味検索のモデルを読み込めません: {error}")
            return False
        self._load_cache()
        self.state = "使用中"
        return True

    @staticmethod
    def _load_model(SentenceTransformer):
        """まずローカルのキャッシュだけで読む(通信ゼロ。毎回の起動でHugging Faceに問い合わせない=レート制限に当たらない)。
        キャッシュが無いときだけ、待ち時間を増やしながら数回ダウンロードを試す(429対策)。"""
        try:
            return SentenceTransformer(vec_model_name, device="cpu", local_files_only=True)
        except TypeError:
            pass    # 古いsentence-transformersはlocal_files_onlyを受け付けない → 通常の読み込みへ
        except Exception:
            pass    # キャッシュが無い → ダウンロードへ
        last = None
        for i in range(vec_dl_retries):
            try:
                if i:
                    wait = min(vec_dl_wait * (2 ** (i - 1)), 600) + random.uniform(0, 3)
                    print(f"Hugging Faceが混み合っています。{wait:.0f}秒後に再試行します({i}/{vec_dl_retries - 1})")
                    time.sleep(wait)    # loadは別スレッドで呼ばれるのでBotは止まらない
                return SentenceTransformer(vec_model_name, device="cpu")
            except Exception as error:
                last = error
                text = f"{type(error).__name__} {error}"
                if not any(w in text for w in ("429", "Too Many Requests", "rate limit", "Rate limit", "ConnectionError", "Timeout", "503", "502")):
                    break    # レート制限や通信エラー以外(モデル名の誤りなど)は待っても直らない
        raise last

    def _load_cache(self):
        try:
            if not self.path.is_file():
                return
            data = np.load(self.path, allow_pickle=False)
            if str(data["model"].item()) != vec_model_name + "|" + vec_prefix + "|s2":
                return
            items = json.loads(str(data["meta"].item()))
            vecs = data["vecs"].astype(np.float32)
            if vecs.ndim == 2 and len(items) == len(vecs):
                with self.lock:
                    self.items, self.mat = items, vecs
                    self.keys = {self.key_of(it) for it in items}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    def sync(self, sm):
        """今ある知識・会話に合わせて、足りないベクトルを作り、不要なものを消す(別スレッドで呼ぶ)。"""
        if self.model is None:
            return
        wanted = {}
        for f in list(sm.facts):
            it = {"k": "f", "t": f["t"]}
            wanted[self.key_of(it)] = it
        for t in list(sm.turns):
            it = {"k": "t", "t": t["q"], "a": t["a"]}
            wanted[self.key_of(it)] = it
        seeds, _, qas = load_train()    # train.jsonの追加学習(基礎文・質問と返答の組)
        for text, weight, core in seeds:
            it = {"k": "s", "t": text, "w": weight}
            wanted[self.key_of(it)] = it
        for q, a, weight in qas:
            it = {"k": "t", "t": q, "a": a, "w": weight}
            wanted[self.key_of(it)] = it
        self.train_mtime = train_mtime()
        with self.lock:
            self._flush()
            if self.mat is not None:
                keep = [i for i, it in enumerate(self.items) if self.key_of(it) in wanted]
                if len(keep) != len(self.items):
                    self.items = [self.items[i] for i in keep]
                    self.mat = self.mat[keep] if keep else None
            self.keys = {self.key_of(it) for it in self.items}
            missing = [it for k, it in wanted.items() if k not in self.keys]
        for i in range(0, len(missing), 512):
            batch = [it for it in missing[i:i + 512] if it["t"].strip()]
            if not batch:
                continue
            vecs = self.encode([it.get("e", it["t"]) for it in batch])
            with self.lock:
                for it, vec in zip(batch, vecs):
                    if self.key_of(it) not in self.keys:
                        self.keys.add(self.key_of(it))
                        self.pending_items.append(it)
                        self.pending_vecs.append(vec)
                        self.dirty += 1
        self.save()

    def save(self):
        if self.model is None:
            return
        if self.todo:
            self.drain()
        with self.lock:
            self._flush()
            items = list(self.items)
            mat = None if self.mat is None else self.mat.astype(np.float16)
            self.dirty = 0
        if mat is None or not items:
            return
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, vecs=mat, meta=np.array(json.dumps(items, ensure_ascii=False)), model=np.array(vec_model_name + "|" + vec_prefix + "|s2"))
        tmp.replace(self.path)

    async def start(self, sm):
        """Bot起動時に裏で呼ぶ。読み込み中もBotは普通に動く(それまでは語の重なりで代用)。"""
        self.sm = sm
        if await asyncio.to_thread(self.load):
            await asyncio.to_thread(self.sync, sm)
            print(f"意味検索: 使用中({len(self.items)}件)")
            asyncio.create_task(self.pump())
        else:
            print(f"意味検索: {self.state}。語の重なりで代用します")

    def add(self, kind, text, answer=""):
        """学習した知識・会話を1件キューに積む(エンコードは裏のpumpがまとめてやる。イベントループを止めない)。"""
        if not self.ready:
            return
        text = str(text or "").strip()
        if not text:
            return
        it = {"k": kind, "t": text}
        if answer:
            it["a"] = answer
        with self.lock:
            if self.key_of(it) in self.keys:
                return
            self.keys.add(self.key_of(it))
            self.todo.append(it)

    def drain(self):
        """キューに溜まった分をまとめてエンコードして取り込む(別スレッドで呼ぶ)。"""
        with self.lock:
            batch, self.todo = self.todo, []
        for i in range(0, len(batch), 64):
            part = batch[i:i + 64]
            vecs = self.encode([it["t"] for it in part])
            with self.lock:
                for it, vec in zip(part, vecs):
                    self.pending_items.append(it)
                    self.pending_vecs.append(vec)
                    self.dirty += 1

    async def pump(self):
        while True:
            await asyncio.sleep(3)
            try:
                if train_mtime() != self.train_mtime and getattr(self, "sm", None) is not None:
                    await asyncio.to_thread(self.sync, self.sm)     # train.jsonが更新されたら追加学習し直す
                if self.todo:
                    await asyncio.to_thread(self.drain)
                if self.dirty >= 200:
                    await asyncio.to_thread(self.save)
            except Exception as error:
                print(f"意味検索の取り込みエラー: {error}")

    def discard(self, keys):
        """消した知識をインデックスからも外す。"""
        if self.model is None or not keys:
            return
        with self.lock:
            self._flush()
            if self.mat is None:
                return
            keep = [i for i, it in enumerate(self.items) if self.key_of(it) not in keys]
            if len(keep) != len(self.items):
                self.items = [self.items[i] for i in keep]
                self.mat = self.mat[keep] if keep else None
                self.dirty += 1
            self.keys -= set(keys)

    def search(self, query, per_kind=6):
        """質問に意味が近い知識(f)と会話(t)を返す。使えないときは None。{"f": [(類似度, item)], "t": [...]}"""
        if not self.ready:
            return None
        query = str(query or "").strip()
        out = {"f": [], "t": [], "s": []}
        if not query:
            return out
        q = self.encode_query(query)
        qn = SharedMemory.normalize(query)
        with self.lock:
            self._flush()
            if self.mat is None or not self.items:
                return out
            sims = self.mat @ q
            order = np.argsort(-sims)[:per_kind * 12]
            for i in order:
                it = self.items[int(i)]
                if len(out[it["k"]]) >= per_kind or (SharedMemory.normalize(it["t"]) == qn and "w" not in it):   # 自分の発言そのものは除く。ただしtrain.json(wつき)の質問は完全一致が本命
                    continue
                out[it["k"]].append((float(sims[int(i)]), it))
        return out

    def hints(self, query, context=None):
        """生成のヒントにする文を返す。意味が十分近いものだけ。意味検索が使えないときは None。"""
        text = str(query or "").strip()
        if context and len(SharedMemory.normalize(text)) <= 8:
            text = f"{context} {text}"
        found = self.search(text)
        if found is None:
            return None
        cands = [(s, it.get("a", "")) for s, it in found["t"] if s >= vec_turn_min and it.get("a")]
        cands += [(s, it["t"]) for s, it in found["f"] if s >= vec_fact_min]
        if not cands:
            return []
        cands.sort(key=lambda x: x[0], reverse=True)
        top = cands[0][0]
        hints = []
        for score, hint in cands:
            if score >= top - vec_margin and hint not in hints:
                hints.append(hint)
            if len(hints) >= vec_max_hints:
                break
        return hints


vec_index = VecIndex()

recent_replies = defaultdict(lambda: deque(maxlen=20))
nuun_gobi = ("ぬーん", "ぬーん", "ぬーん", "ぬん")


def _moji_kind(ch):
    if re.match(r"[一-龥々〆ヵヶ]", ch):
        return "k"      # 漢字
    if re.match(r"[ぁ-ん]", ch):
        return "h"      # ひらがな
    if re.match(r"[ァ-ヴ]", ch):
        return "a"      # カタカナ
    if re.match(r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]", ch):
        return "l"      # 英数字
    return "o"          # 記号・空白・絵文字など


nuun_joshi = "はがをにでともへや"


_tok = None
_tok_ng = False
nuun_jiritsu = {"名詞", "動詞", "形容詞", "副詞", "接続詞", "感動詞", "接頭詞"}


def get_tokenizer():
    """形態素解析器(Janome: pip install janome。辞書内蔵・アカウント不要・LLMではない)。無ければ None。"""
    global _tok, _tok_ng
    if _tok is None and not _tok_ng:
        try:
            from janome.tokenizer import Tokenizer
            _tok = Tokenizer()
        except Exception as error:
            _tok_ng = True
            print(f"形態素解析なし(pip install janome で単語境界が正確になる): {error}")
    return _tok


def nuun_kugiri_keitaiso(text, end):
    """形態素解析で、文節の切れ目(自立語の直前など)だけを返す。使えないとき・文字数が合わないときは None。"""
    tok = get_tokenizer()
    if tok is None:
        return None
    try:
        toks = [(t.surface, t.part_of_speech.split(",")) for t in tok.tokenize(text)]
    except Exception:
        return None
    if sum(len(sf) for sf, _ in toks) != len(text):
        return None
    cuts = []
    pos = 0
    prev = None
    for surf, pp in toks:
        if prev is not None and 0 < pos < end:
            p1 = pp[0]
            p2 = pp[1] if len(pp) > 1 else ""
            q1 = prev[0]
            head = p1 in nuun_jiritsu and p2 not in ("非自立", "接尾")
            if q1 == "記号" and p1 != "記号":
                head = True
            if q1 in ("接頭詞", "連体詞"):
                head = False        # お金・この人 のように次の語に続く
            elif q1 == p1 and q1 in ("名詞", "動詞"):
                head = False        # 複合語(天気予報・話し合う)の途中
            if head:
                cuts.append(pos)
        pos += len(surf)
        prev = (pp[0],)
    return cuts


def nuun_kugiri(text, end):
    """単語の途中ではない挿入位置(文字の間の番号)を返す。形態素解析はせず、文字種の切れ目で判断する。
    ・記号/空白/英数字/カタカナの前後 ・ひらがな→漢字の切れ目 ・漢字/カタカナ/英数字に続く助詞の後ろ ・「を」の後ろ。
    ひらがなだけが続く所は単語の途中かもしれないので切らない。<...>(カスタム絵文字など)の中も避ける。"""
    kinds = []
    for i, ch in enumerate(text):
        k = _moji_kind(ch)
        if ch in "ーｰ" and i:
            k = kinds[-1]       # 長音は直前の文字種に従う
        kinds.append(k)
    inside = set()
    for m in re.finditer(r"<[^>\n]*>", text):
        inside.update(range(m.start() + 1, m.end()))
    morph = nuun_kugiri_keitaiso(text, end)
    if morph is not None:
        # 形態素解析が使えるときは、その切れ目だけを使う(英数字の途中・<...>の中は除く)。
        return [i for i in morph if i not in inside and not (kinds[i - 1] == kinds[i] == "l")]
    out = []
    for i in range(1, end):
        if i in inside:
            continue
        a, b = kinds[i - 1], kinds[i]
        ok = False
        if a == b and a in ("l", "o"):
            ok = False          # 英数字の途中・記号の連続の途中は切らない
        elif "o" in (a, b) or "l" in (a, b):
            ok = True
        elif a == "h" and b == "k":
            ok = True
        elif "a" in (a, b) and a != b:
            ok = True
        elif text[i - 1] == "を":
            ok = True
        elif text[i - 1] in nuun_joshi and i >= 2 and kinds[i - 2] in ("k", "a", "l"):
            ok = True
        if ok:
            out.append(i)
    return out


def nuunka(text):
    """ぬーん必須: 返信に「ぬーん」(ぬ/ヌ/Nで始まりん/ン/Nで終わる8文字以内のことば)が無ければ、
    文の中の、単語の途中ではない位置にランダムで差し込む。どこにも切れ目が無い短い文は文頭か文末に付ける。"""
    text = str(text).strip()
    if nuunre.search(text):
        return text
    end = len(re.sub(r"[。！？!?…~〜]+$", "", text))
    cuts = nuun_kugiri(text, end)
    pos = random.choice(cuts) if cuts else random.choice([0, end])
    return text[:pos] + "ぬーん" + text[pos:]


_train_cache = {"mtime": None, "seeds": [], "fallbacks": [], "qas": []}
nuun_koe_moji = set("ぬヌぅぉんンーｰ〜~ぁぃぇっッ！!？?…・、。nNuU 　")


def train_mtime():
    try:
        return trainingfile.stat().st_mtime
    except OSError:
        return None


def load_train():
    """train.jsonのgroupsを読む。更新されたら読み直す。(seeds, fallbacks, qas) を返す。
    ・{"q": "質問" or ["質問",...], "text": "返答"} → qas: 質問と返答の組(会話として学習する)
    ・それ以外の文(おはようぬーん、ぬーーーん等。ぬーん系も含めて全部同じ扱い) → seeds: 文そのままで意味検索する
    weight / repeat / enabled / group_selection / item_selection は、従来の学習設定と同じ重みの出し方。"""
    mtime = train_mtime()
    if mtime is None:
        return [], [], []
    if _train_cache["mtime"] == mtime:
        return _train_cache["seeds"], _train_cache["fallbacks"], _train_cache["qas"]
    seeds, fallbacks, qas = {}, {}, []
    try:
        data = json.loads(trainingfile.read_text("utf-8"))
        st = data.get("settings", {})
        default_weight = max(1, int(st.get("default_weight", 2)))
        default_repeat = max(1, min(int(st.get("repeat", 1)), 50))
        group_mode = st.get("group_selection", "weight")
        item_mode = st.get("item_selection", "weight")
        for group in data.get("groups", []):
            if not isinstance(group, dict) or not group.get("enabled", True):
                continue
            group_weight = max(1, int(group.get("weight", default_weight)))
            group_fallback = bool(group.get("fallback", True))      # 近い会話が無いときの雑談に使うか
            group_repeat = max(1, min(int(group.get("repeat", default_repeat)), 50))
            if group_mode == "weight":
                group_repeat *= group_weight
            for item in group.get("items", []):
                questions = []
                item_fallback = group_fallback
                if isinstance(item, str):
                    text, item_weight, item_repeat = item, group_weight, group_repeat
                elif isinstance(item, dict) and item.get("enabled", True):
                    text = item.get("text", "")
                    item_weight = max(1, int(item.get("weight", group_weight)))
                    item_repeat = max(1, min(int(item.get("repeat", group_repeat)), 50))
                    item_fallback = bool(item.get("fallback", group_fallback))
                    q = item.get("q", [])
                    questions = [q] if isinstance(q, str) else [x for x in q if isinstance(x, str)]
                else:
                    continue
                text = str(text or "").strip()
                if not text:
                    continue
                weight = item_weight * item_repeat if item_mode == "weight" else item_weight
                qs = [x.strip() for x in questions if x.strip()]
                if qs:
                    qas.extend((x, text, weight) for x in qs)
                    continue
                # ぬーん系(鳴き声だけの文も、ぬーんを含む文も)を区別せず、文そのままをベクトルにして学習する。
                seeds[text] = (seeds[text][0] + weight, text) if text in seeds else (weight, text)
                if item_fallback:
                    fallbacks[text] = fallbacks.get(text, 0) + weight
    except (json.JSONDecodeError, OSError, TypeError, ValueError, AttributeError):
        seeds, fallbacks, qas = {}, {}, []
    seeds = [(t, w, core) for t, (w, core) in seeds.items()]
    fallbacks = list(fallbacks.items())
    _train_cache.update(mtime=mtime, seeds=seeds, fallbacks=fallbacks, qas=qas)
    return seeds, fallbacks, qas


def load_seeds():
    """意味検索に当たらなかったときの雑談用: fallbackがtrue(省略時true)のグループの文だけを、重み付きで使う。
    挨拶・返事・感情のような場面に結びついた文はfallback:falseにして、意味検索で近いときだけ使わせる。"""
    return load_train()[1]


nuun_kihon_katachi = [
    "{}が{}している", "{}は{}だ", "今日も{}", "{}かもしれない", "なんとなく{}", "{}の気配がする",
    "{}すぎて{}", "{}って言った？", "{}について考えている", "{}がそこにいる",
]


def nuun_kotoba():
    """意味のないぬーん語を作る。train.jsonの鳴き声(ぬーーーん等)があればそこから、無ければ文字を組み合わせて。"""
    voices = [t for t, _ in load_seeds() if nuunre.fullmatch(t)]
    if voices and random.random() < 0.5:
        return random.choice(voices)
    return "ぬ" + "".join(random.choice("ーーーぅぉ") for _ in range(random.randint(1, 4))) + "ん"


def nuun_seisei():
    """近い会話も知識も無いときの予備: 意味のないぬーんの文を、決まった型に当てはめて作る(LLMも機械学習も使わない)。
    型は train.json の予備用の文(fallback:true。ぬーん語を穴にしたもの)と内蔵の型。
    穴は毎回ちがうぬーん語で埋め、2つの型の前半と後半をつなぐこともある。"""
    katachi = []
    for t, w in load_seeds():
        parts = nuunre.split(t)
        if len(parts) >= 2 and len("".join(parts).strip(" 、。！？!?…~〜")) >= 2:
            katachi.append((parts, math.sqrt(w)))
    for k in nuun_kihon_katachi:
        katachi.append((k.split("{}"), 3.0))

    def fill(parts):
        out = parts[0]
        for piece in parts[1:]:
            out += nuun_kotoba() + piece
        return out

    def pick():
        return fill(random.choices([k for k, _ in katachi], weights=[w for _, w in katachi], k=1)[0])

    out = pick()
    if random.random() < 0.4:
        a = re.split(r"(?<=[、。！？!?])", out)[0].rstrip("。")
        b = re.split(r"(?<=[、。！？!?])", pick())[-1]
        if len(a) >= 2 and len(b) >= 2:
            out = (a if a[-1] in "、？！!?" else a + "、") + b
    out = out.strip()
    return out if nuunre.search(out) and len(out) <= 80 else nuun_kotoba()


def kotaeru_sync(message_text, context_text, is_question, cid):
    """意味検索だけで返信を選ぶ。文章は生成しない・LLMは使わない。
    1) 意味が近い過去の質問に続いた返答(会話) 2) 質問のときは、近い知識。
    今の発言そのものを先に探し、何も無いときだけ直前の発言を足して探し直す(前の発言に引きずられて1つ遅れた返事になるのを防ぐ)。
    短いときは、近い候補をもう1〜2件つなげて長くする。近い候補が無ければ予備のぬーん文。"""
    ng = {SharedMemory.normalize(message_text), SharedMemory.normalize("ぬーんをやりるぬーん")}
    ng |= {SharedMemory.normalize(x) for x in recent_replies[cid]}

    def ok(x):
        x = str(x or "").strip()
        n = SharedMemory.normalize(x)
        return len(n) >= 2 and len(x) <= 200 and n not in ng and not re.search(r"https?://|<@|<#|@everyone|@here", x)

    def finish(out, pool):
        """短ければ、同じくらい近い別の候補をつなげる。"""
        out = out.strip()
        used = {SharedMemory.normalize(out)}
        extra = [c for _, c in pool if SharedMemory.normalize(c) not in used]
        random.shuffle(extra)
        for c in extra:
            if len(SharedMemory.normalize(out)) >= reply_min_chars:
                break
            c = c.strip()
            n = SharedMemory.normalize(c)
            if n in used or len(out) + len(c) + 1 > reply_max_chars:
                continue
            used.add(n)
            out = out + ("" if out[-1:] in "。！？!?…" else "。") + c
        recent_replies[cid].append(out)
        return nuunka(out)

    # 質問が教えられた知識の話題と一致するとき(「ふにゃは？」「ふにゃってなんなの」←「ふにゃはすらいむ」)は、類似度に頼らず確実に拾う。
    lex = []
    if is_question or shared_memory.chishiki_wadai(message_text)[1]:
        for full, _ in shared_memory.chishiki_zenpou(message_text, force=True, limit=3):
            if ok(full):
                lex.append((0.99, full))

    def pick_from(found):
        c = [(s, it["a"]) for s, it in found["t"] if s >= vec_turn_min and ok(it.get("a"))]
        c += lex
        if not c and is_question:
            c = [(s, it["t"]) for s, it in found["f"] if s >= vec_fact_min and ok(it["t"])]
        return c

    if vec_index.ready and vec_index.todo:
        vec_index.drain()           # 直前に学習した分を、裏のポンプ(3秒おき)を待たずに今すぐ検索に反映する
    found = vec_index.search(message_text)
    if found is not None:
        cands = pick_from(found)
        if not cands and context_text and len(SharedMemory.normalize(message_text)) <= 8:
            # 「うん」「それは？」のような短い発言は、今の発言だけでは分からないので、直前の発言を足して探し直す。
            found = vec_index.search(f"{context_text} {message_text}") or found
            cands = pick_from(found)
    else:
        # 意味検索モデルが読み込み中・使えないときは、語の重なりで探す(これも文章は生成しない)。
        raw = [f["t"] for f in shared_memory.chishiki_sagasu(message_text)] if is_question else []
        raw.append(shared_memory.recall(message_text, context=context_text))
        cands = lex + [(1.0, x) for x in raw if ok(x)]
    if not cands and found is not None:
        # 近い会話・知識が無ければ、train.jsonの基礎文のうち話題が近いものを選ぶ。
        seeded = [(s_, it["t"], it.get("w", 1)) for s_, it in found["s"] if s_ >= vec_seed_min and ok(it["t"])]
        if seeded:
            top_s = max(x[0] for x in seeded)
            pool_s = [x for x in seeded if x[0] >= top_s - vec_margin]
            out = random.choices([x[1] for x in pool_s], weights=[math.sqrt(x[2]) * math.exp((x[0] - top_s) / 0.03) for x in pool_s], k=1)[0]
            return finish(out, [(x[0], x[1]) for x in pool_s])
    if not cands:
        # 何も近くないときは、意味のないぬーんの文を作って返す。
        for _ in range(6):
            out = nuun_seisei()
            if ok(out):
                pool_n = []
                if len(SharedMemory.normalize(out)) < reply_min_chars:
                    pool_n = [(0, nuun_seisei()) for _ in range(2)]
                return finish(out, pool_n)
        return None
    top = max(c[0] for c in cands)
    pool = [(s, c) for s, c in cands if s >= top - vec_margin]
    # 類似度が高い候補ほど選ばれやすい(同点圏ではばらける)。
    out = random.choices([c for _, c in pool], weights=[math.exp((s - top) / 0.03) for s, _ in pool], k=1)[0]
    # 短いときにつなげる候補は、マージンの外でも少し近いものまで広げる。
    wide = [(s, c) for s, c in cands if s >= top - 0.15]
    return finish(out, sorted(wide, key=lambda x: x[0], reverse=True))


async def kotaeru(message_text, context_text, is_question, cid):
    # エンコードはCPUを使うので、イベントループを止めないよう別スレッドで行う。
    return await asyncio.to_thread(kotaeru_sync, message_text, context_text, is_question, cid)


class NuunGakushu:
    def __init__(self):
        self.ngram = 3
        self.max_learn_length = 300
        self.generation_max = 180
        self.dirty = 0
        self.models = {}                            
        self.legacy = None                                
        self.settings = {}
        self.recent = []
        self.recent_a = []
        self.base_models = {}
        self.model = self.sinki()
        self.load()

    def sinki(self):
        return {
            "total": 0, "chars": {}, "starts": {}, "ends": {},
            "sentences": [], "talk": [], "chars2": {}, "chars5": {},
            "kioku": {"df": {}, "cooc": {}, "reply": {}, "short": {}, "aizuchi": {}, "n": 0, "pairs": 0},
        }

    def seiri(self, m):
        for name in ("chars", "starts", "ends", "chars2", "chars5"):
            m.setdefault(name, {})
        m.setdefault("total", 0)
        m.setdefault("sentences", [])
        m.setdefault("talk", [])
        m.pop("taiwa", None)
        k = m.setdefault("kioku", {})
        for name in ("df", "cooc", "reply", "short", "aizuchi", "facts"):
            k.setdefault(name, {})
        k.setdefault("n", 0)
        k.setdefault("pairs", 0)
        if not m["chars2"] or not m["chars5"]:
            for sentence in m["sentences"] + m["talk"]:
                for i in range(len(sentence) - 2):
                    self._add(m["chars2"], sentence[i:i + 2], sentence[i + 2], 1)
                for i in range(len(sentence) - 5):
                    self._add(m["chars5"], sentence[i:i + 5], sentence[i + 5], 1)

    def _add_sentence_to_base(self, base, sentence):
        text = self.clean(sentence)
        if len(text) < self.ngram + 1:
            return
        self._add(base["starts"], text[:self.ngram], text[0], 1)
        for i in range(len(text) - self.ngram):
            self._add(base["chars"], text[i:i + self.ngram], text[i + self.ngram], 1)
        for i in range(len(text) - 2):
            self._add(base["chars2"], text[i:i + 2], text[i + 2], 1)
        for i in range(len(text) - 5):
            self._add(base["chars5"], text[i:i + 5], text[i + 5], 1)
        self._add(base["ends"], text[-self.ngram:], text[-1], 1)
        words = self.word_tokens(text)
        if len(words) < 3:
            return
        base["word_starts"].append(words[:2])
        base["word_ends"].append(words[-1])
        topic_tokens = frozenset(token for token in words if len(token) >= 2 and not nuunre.fullmatch(token))
        if topic_tokens:
            entry = (tuple(words[:2]), topic_tokens)
            for token in topic_tokens:
                base.setdefault("topic_index", {}).setdefault(token, []).append(entry)
        for i in range(1, len(words)):
            self._add(base["word_next"], self.word_key(words[i - 1:i]), words[i], 1)

    def base_model(self, gid=None):
        if gid is None:
            gid = next((k for k, v in self.models.items() if v is self.model), None)
        gid = str(gid or "0")
        if gid in self.base_models:
            return self.base_models[gid]
        m = self.model
        base = {"chars": {}, "starts": {}, "ends": {}, "chars2": {}, "chars5": {},
                "word_next": {}, "word_starts": [], "word_ends": [], "topic_index": {}}
        for sentence in list(m.get("sentences", [])) + list(m.get("talk", [])):
            text = self.clean(sentence)
            if self.clean(text).replace(" ", "") == "ぬーんをやりるぬーん":
                continue
            if len(text) < self.ngram + 1:
                continue
            self._add(base["starts"], text[:self.ngram], text[0], 1)
            for i in range(len(text) - self.ngram):
                self._add(base["chars"], text[i:i + self.ngram], text[i + self.ngram], 1)
            for i in range(len(text) - 2):
                self._add(base["chars2"], text[i:i + 2], text[i + 2], 1)
            for i in range(len(text) - 5):
                self._add(base["chars5"], text[i:i + 5], text[i + 5], 1)
            self._add(base["ends"], text[-self.ngram:], text[-1], 1)
            words = self.word_tokens(text)
            if len(words) >= 3:
                base["word_starts"].append(words[:2])
                base["word_ends"].append(words[-1])
                topic_tokens = frozenset(token for token in words if len(token) >= 2 and not nuunre.fullmatch(token))
                if topic_tokens:
                    entry = (tuple(words[:2]), topic_tokens)
                    for token in topic_tokens:
                        base["topic_index"].setdefault(token, []).append(entry)
                for i in range(1, len(words)):
                    key = self.word_key(words[i - 1:i])
                    self._add(base["word_next"], key, words[i], 1)
        self.base_models[gid] = base
        return base

    def word_tokens(self, text):
        # 「ぬーん」を前後のひらがなに吸収させず、ひとまとまりとして扱う。
        text = self.clean(text)
        parts = re.split(r"(ぬーん|ぬ〜ん|ぬん|ヌーン|ヌン|nu+n)", text, flags=re.I)
        tokens = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"ぬーん|ぬ〜ん|ぬん|ヌーン|ヌン|nu+n", part, re.I):
                tokens.append(part)
            else:
                tokens.extend(re.findall(r"[一-龥々]+[ぁ-ん]*|[ぁ-ん]+|[ァ-ヴー]+|[A-Za-z0-9]+|[、。！？!?…~〜]+|[^\s]", part))
        return tokens

    def word_key(self, words):
        return "\x1f".join(words)

    def use(self, gid):
        """このサーバーの学習内容に切り替える。初めてのサーバーなら、train.jsonの基礎だけ覚えた状態で作る。"""
        gid = str(gid)
        m = self.models.get(gid)
        if m is None:
            m = self.models[gid] = self.sinki()
            self.model = m
            self.learn_settings()
        self.model = m
        self.base_model(gid)
        return m

    def kidou(self):
        self.yomu_settings()
        for gid in list(self.models):
            self.model = self.models[gid]
            self.learn_settings()
        for gid in self.models:
            self.model = self.models[gid]
            self.base_model(gid)
        print(f"学習データ: {len(self.models)}サーバー分")

    def load(self):
        data = {}
        if modelfile.is_file():
            try:
                data = json.loads(modelfile.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        if isinstance(data.get("saba"), dict):
            self.models = data["saba"]
            self.legacy = data.get("kyotsu")
        elif data:
            self.legacy = data
        for m in self.models.values():
            self.seiri(m)

    def save(self):
        cur = self.model
        for m in self.models.values():
            self.model = m
            self.prune()
        self.model = cur
        data = {"version": 3, "saba": self.models}
        if self.legacy:
            data["kyotsu"] = self.legacy
        tmp = modelfile.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), "utf-8")
        tmp.replace(modelfile)
        self.dirty = 0

    def clean(self, text):
        text = str(text or "")
        text = re.sub(r"<@!?\d+>", "", text)
        text = re.sub(r"<#\d+>", "", text)
        text = re.sub(r"<@&\d+>", "", text)
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"https?://\S+", "", text)
        text = text.replace("\\n", "").replace("\\r", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:self.max_learn_length]

    def _add(self, table, key, value, weight=1):
        table.setdefault(key, {})
        table[key][value] = table[key].get(value, 0) + weight

    def learn(self, text, weight=1, talk=False):
        text = self.clean(text)
        if len(text) < self.ngram + 1:
            return

        weight = max(1, min(int(weight), 100))
        self.model["total"] += weight
        self.dirty += 1

        self._add(self.model["starts"], text[:self.ngram], text[0], weight)

        for i in range(len(text) - self.ngram):
            self._add(
                self.model["chars"],
                text[i:i + self.ngram],
                text[i + self.ngram],
                weight
            )

        for i in range(len(text) - 2):
            self._add(self.model["chars2"], text[i:i + 2], text[i + 2], weight)

        for i in range(len(text) - 5):
            self._add(self.model["chars5"], text[i:i + 5], text[i + 5], weight)

        self._add(self.model["ends"], text[-self.ngram:], text[-1], weight)

        sentences = self.model["talk"] if talk else self.model["sentences"]
        if text not in sentences:
            sentences.append(text)
            limit = 2000 if talk else 5000
            if len(sentences) > limit:
                del sentences[:-limit]
            gid = str(next((k for k, value in self.models.items() if value is self.model), "0"))
            base = self.base_models.get(gid)
            if base is not None:
                self._add_sentence_to_base(base, text)

    def yomu_settings(self):
        if not trainingfile.is_file():
            return None
        try:
            data = json.loads(trainingfile.read_text("utf-8"))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None

        self.settings = data.get("settings", {})
        self.ngram = max(2, min(int(self.settings.get("ngram", 3)), 6))
        self.max_learn_length = max(
            self.ngram + 1,
            min(int(self.settings.get("max_learn_length", 300)), 1000)
        )
        self.generation_max = max(
            20,
            min(int(self.settings.get("generation_max", 180)), 500)
        )
                                                   
        self.maxweak = max(0, min(int(self.settings.get("weak_join_max", 2)), 20))
        return data

    def learn_settings(self):
        data = self.yomu_settings()
        if data is None:
            return

                                                        
        digest = hashlib.sha256(
            json.dumps(data.get("groups", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if "training_hash_v2" not in self.model and (
            "training_hash" in self.model or self.model.get("training_learned")
        ):
            self.model["training_hash_v2"] = digest
            return
        if self.model.get("training_hash_v2") == digest:
            return

        default_weight = max(1, int(self.settings.get("default_weight", 2)))
        default_repeat = max(1, min(int(self.settings.get("repeat", 1)), 50))
        group_mode = self.settings.get("group_selection", "weight")
        item_mode = self.settings.get("item_selection", "weight")

        for group in data.get("groups", []):
            if not isinstance(group, dict) or not group.get("enabled", True):
                continue

            group_weight = max(1, int(group.get("weight", default_weight)))
            group_repeat = max(1, min(int(group.get("repeat", default_repeat)), 50))

            if group_mode == "weight":
                group_repeat *= group_weight

            for item in group.get("items", []):
                if isinstance(item, str):
                    text = item
                    item_weight = group_weight
                    item_repeat = group_repeat
                elif isinstance(item, dict) and item.get("enabled", True):
                    text = item.get("text", "")
                    item_weight = max(1, int(item.get("weight", group_weight)))
                    item_repeat = max(1, min(int(item.get("repeat", group_repeat)), 50))
                else:
                    continue

                if not text:
                    continue

                total_weight = (
                    item_weight * item_repeat
                    if item_mode == "weight"
                    else item_weight
                )
                self.learn(text, total_weight)

        self.model["training_hash_v2"] = digest
        self.save()

    def keyword_score(self, message, rule):
        message = self.clean(message)
        keywords = rule.get("kiwado", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        if not keywords:
            return 0.18

        hits = sum(1 for keyword in keywords if keyword and keyword in message)
        return min(1.0, 0.42 + hits * 0.16) if hits else 0.0

    def nuun_score(self, message):
        text = self.clean(message)
        if len(text) <= self.ngram:
            return 0.0

        known = 0
        total = 0

        for i in range(len(text) - self.ngram):
            total += 1
            if text[i:i + self.ngram] in self.model["chars"]:
                known += 1

        return known / total if total else 0.0

    def reply_probability(self, message, rule):
        try:
            base = max(0.0, min(100.0, float(rule.get("ritsu", 0)))) / 100.0
        except (TypeError, ValueError, AttributeError):
            return 0.0

        if base <= 0:
            return 0.0
        if base >= 1:
            return 1.0

        key_score = self.keyword_score(message, rule)
        style_score = self.nuun_score(message)

        probability = base * (0.72 + key_score * 0.20 + style_score * 0.08)

        if key_score > 0:
            probability = max(probability, base * 0.55)

        return max(0.0, min(1.0, probability))

    def weighted_choice(self, values):
        if not values:
            return None

        total = sum(max(0, float(value)) for value in values.values())
        if total <= 0:
            return random.choice(list(values))

        point = random.random() * total

        for value, weight in values.items():
            point -= max(0, float(weight))
            if point <= 0:
                return value

        return next(iter(values))

                             
    def kioku(self):
        return self.model["kioku"]

    def tokens(self, text):
        out = []
        for t in tokenre.findall(self.clean(text)):
            if t in tomeru or len(t) > 8 or nuunre.search(t) or t in out:
                continue
            out.append(t)
        return out[:8]

    def idf(self, t):
        k = self.kioku()
        return math.log((k["n"] + 2) / (k["df"].get(t, 0) + 1)) + 1

    def learn_kioku(self, text):
        """同じ発言に出てきた言葉同士を結びつける。"""
        k = self.kioku()
        toks = self.tokens(text)
        if not toks:
            return
        k["n"] += 1
        self.dirty += 1
        for t in toks:
            k["df"][t] = k["df"].get(t, 0) + 1
        for a in toks:
            for b in toks:
                if a != b:
                    self._add(k["cooc"], a, b)

    def learn_fact(self, text):
        """互換用。事実を主語と答えの固定ペアにはせず、共有コーパスに発言全体を保存する。"""
        text = self.clean(text)
        if len(text) < 2:
            return
        shared_memory.learn(text)

    def recall_fact(self, question, gid=0):
        """共有コーパスから質問文に続く既出の文章を検索する。"""
        return shared_memory.recall(question)

    def learn_pair(self, q, a):
        """qに対してaと返した、という流れを学習する(返信 or 直後の別人の発言)。"""
        k = self.kioku()
        tq = self.tokens(q)[:6]
        ta = self.tokens(a)[:6]
        for x in tq:
            for y in ta:
                if x != y:
                    self._add(k["reply"], x, y)

        short = self.clean(a)
        if (
            1 <= len(short) <= 10
            and re.search(r"[ぁ-んァ-ヴ一-龥A-Za-z0-9]", short)
            and not nuunre.search(short)
            and not self.too_close(short, q)
        ):
            for x in tq:
                self._add(k["short"], x, short)
            k["aizuchi"][short] = k["aizuchi"].get(short, 0) + 1

        k["pairs"] += 1
        self.dirty += 1

    def prune(self):
        k = self.kioku()
        for name in ("cooc", "reply", "short"):
            table = k[name]
            for key in list(table):
                inner = table[key]
                if len(inner) > 60:
                    table[key] = dict(sorted(inner.items(), key=lambda kv: -kv[1])[:40])
            if len(table) > 6000:
                for key in [x for x, v in table.items() if sum(v.values()) <= 1]:
                    del table[key]
        if len(k["df"]) > 20000:
            for key in [x for x, v in k["df"].items() if v <= 1]:
                del k["df"][key]
        if len(k["aizuchi"]) > 3000:
            for key in [x for x, v in k["aizuchi"].items() if v <= 1]:
                del k["aizuchi"][key]

    def assoc_weights(self, t):
        k = self.kioku()
        w = {}
        for y, c in k["reply"].get(t, {}).items():
            w[y] = w.get(y, 0) + c * 3
        for y, c in k["cooc"].get(t, {}).items():
            w[y] = w.get(y, 0) + c
        return w

    def hentou(self, message, gid=0, hints=None):
        self.use(gid)
        out = self.generate(message, hints=hints)
        if not out and hints:
            # ヒントに寄せた文が作れなかったときは、ヒントなしの通常生成にする(保存文は出さない)。
            out = self.generate(message)
        if out:
            self.recent = (self.recent + [out])[-20:]
        return out

    def topic_chars(self, seed):
        if self.kioku()["n"] < 20:
            return set()
        toks = sorted(self.tokens(seed), key=lambda t: -self.idf(t))[:2]
        return set("".join(toks))

    def bias(self, candidates, boost, factor=4):
        if not boost:
            return candidates
        return {c: (w * factor if c in boost else w) for c, w in candidates.items()}

    def hint_words(self, hints):
        """ヒント文から内容語だけを取り出す。文そのものは生成器に渡さない。"""
        words = set()
        for hint in hints or ():
            for token in self.word_tokens(self.clean(hint)):
                if len(token) >= 2 and not nuunre.fullmatch(token):
                    words.add(token)
        return words

    def copied_from_hints(self, out, hints):
        """ヒント文の長い連続コピー(または大半が同じ文)を検出する。"""
        a = re.sub(r"[\s、。！？!?…~〜]", "", self.clean(out))
        if not a:
            return True
        for hint in hints or ():
            b = re.sub(r"[\s、。！？!?…~〜]", "", self.clean(hint))
            if not b:
                continue
            m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
            if m.size >= 10 or m.size >= 0.6 * len(a):
                return True
        return False

    def kioku_setsumei(self, go="", detail=False, gid=0):
        self.use(gid)
        k = self.kioku()
        go = self.clean(go)
        if not go:
            top = sorted(k["df"].items(), key=lambda kv: -kv[1])[:8]
            words = "、".join(f"{w}({c})" for w, c in top) or "まだない"
            aizuchi = sorted(k["aizuchi"].items(), key=lambda kv: -kv[1])[:5]
            ai = "、".join(f"{w}({c})" for w, c in aizuchi if c >= 2) or "まだない"
            return (
                f"このサーバーで覚えた発言: {k['n']}件 / 会話の流れ: {k['pairs']}件\n"
                f"よく出る言葉: {words}\n学んだ相づち: {ai}"
            )

        toks = self.tokens(go)
        x = toks[0] if toks else go
        aw = self.assoc_weights(x)
        lines = []
        if aw:
            rel = sorted(aw.items(), key=lambda kv: -kv[1])[:6]
            shorts = sorted(k["short"].get(x, {}).items(), key=lambda kv: -kv[1])[:4]
            lines.append(f"「{x}」と結びついた言葉: " + "、".join(f"{w}({c})" for w, c in rel))
            if shorts:
                lines.append("返されがちな相づち: " + "、".join(f"{w}({c})" for w, c in shorts))
        else:
            lines.append(f"「{x}」はまだ何も結びついていないぬーん")

        return "\n".join(lines)

    def too_close(self, a, b):
        """曖昧な類似度では弾かず、完全一致か長い原文包含だけを検出する。"""
        a = re.sub(r"\s+", "", self.clean(a))
        b = re.sub(r"\s+", "", self.clean(b))
        if not a or not b:
            return False
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return len(shorter) >= 12 and shorter in longer and len(shorter) / len(longer) >= 0.65

    def pick_sentence(self, seed=""):
        seed = self.clean(seed)

        sentences = list(self.model.get("sentences", []))
        if not sentences:
            return None

        seed = self.clean(seed)
        far = [x for x in sentences if not self.too_close(x, seed)]
        if far:
            sentences = far

                                         
        if len(seed) >= 2:
            grams = list(dict.fromkeys(seed[i:i + 2] for i in range(len(seed) - 1)))[:40]
            scored = []
            for sentence in sentences:
                hit = sum(1 for gram in grams if gram in sentence)
                if hit:
                    scored.append((hit, sentence))
            if scored:
                best = max(hit for hit, _ in scored)
                sentences = [sentence for hit, sentence in scored if hit >= best * 0.6]

        preferred = [
            sentence for sentence in sentences
            if 10 <= len(sentence) <= self.generation_max
        ]

        if preferred:
            sentences = preferred

        return random.choice(sentences)

    def kiri(self, text):
        m = None
        for m in owarire.finditer(text):
            pass
        return bool(m) and m.end() == len(text)

    def tojiru(self, text, tail_only=False):
        """途中で切れた文を、最後のきりのいい所まで戻すか、ぬーんで閉じる。"""
        last = 0
        for m in owarire.finditer(text):
            last = m.end()
        if last >= 6 and not tail_only and nuunre.search(text[:last]):
            return text[:last]

        endings = [
            x for x in self.model.get("sentences", [])
            if len(x) <= 6 and nuunre.fullmatch(x.strip("。！？…~〜 "))
        ]
        tail = random.choice(endings) if endings else "ぬーん"
        return (text.rstrip("。！？ ")[:self.generation_max - len(tail)] + tail)

    def seiritsu(self, out):
        """文として最低限成立しているか。括弧・助詞の重なり・繰り返し・文頭文末を見る。"""
        if not out or len(out) < 3:
            return False
        for a, b in ("「」", "『』", "（）", "()", "[]", "【】"):
            if out.count(a) != out.count(b):
                return False
        particle = "はがをにでとものへ"
        if out[0] in particle + "、。！？)」』）":
            return False
        if re.search(r"[はがをにでとものへ]{2}(?![ぬんー])", out.replace("ぬーん", "")) and re.search(
            r"([はがをにへ])\1", out
        ):
            return False
        if re.search(r"[はがをにでとのへ][、。！？]", out) or re.search(r"、[、。！？]", out):
            return False
        grams = [out[i:i + 4] for i in range(len(out) - 3)]
        if len(grams) != len(set(grams)) and not re.fullmatch(r"[ぬんー！？。…~〜 ]+", out):
            return False
        for seg in re.split(r"[。！？!?…]+", out):
            seg = seg.strip()
            if seg and (len(seg) < 2 or seg[0] in particle):
                return False
        return True

    def maneka(self, out):
        """原文の完全再生と長い連続コピーだけを拒否する。曖昧な類似度判定はしない。"""
        text = re.sub(r"[\s、。！？!?…~〜]", "", self.clean(out))
        if not text:
            return True
        sources = list(self.model.get("sentences", []))[-400:] + list(self.model.get("talk", []))[-200:]
        for source in sources:
            source = re.sub(r"[\s、。！？!?…~〜]", "", self.clean(source))
            if not source:
                continue
            if text == source:
                return True
            shorter, longer = (text, source) if len(text) <= len(source) else (source, text)
            if len(shorter) >= 12 and shorter in longer and len(shorter) / len(longer) >= 0.65:
                return True
        return False

    def generate(self, seed="", length=None, hints=None):
        seed_clean = self.clean(seed)
        for _ in range(40):
            out = self._generate_once(seed_clean, length, hints)
            if not out:
                continue
            if hints and self.copied_from_hints(out, hints):
                continue
            if not self.seiritsu(out):
                continue
            if self.maneka(out):
                continue
            if not self.kiri(out):
                continue
            if not nuunre.search(out):
                continue
            normalized_out = re.sub(r"[\s、。！？!?…~〜]", "", out)
            if any(normalized_out == re.sub(r"[\s、。！？!?…~〜]", "", old) for old in self.recent[-5:]):
                continue
            # Topic overlap is used to choose a seed above; don't make it a hard
            # rejection here, because short/casual messages often share no tokens.
            return out
        return None

    def query_topic_overlap(self, seed, candidate):
        seed_tokens = {x for x in self.word_tokens(seed) if len(x) >= 2 and not nuunre.fullmatch(x)}
        candidate_tokens = {x for x in self.word_tokens(candidate) if len(x) >= 2 and not nuunre.fullmatch(x)}
        if not seed_tokens or not candidate_tokens:
            seed_chars = set(re.sub(r"[\s、。！？!?…~〜]", "", seed))
            candidate_chars = set(re.sub(r"[\s、。！？!?…~〜]", "", candidate))
            return len(seed_chars & candidate_chars) / max(1, len(seed_chars | candidate_chars))
        return len(seed_tokens & candidate_tokens) / max(1, len(seed_tokens | candidate_tokens))

    def _generate_once(self, seed="", length=None, hints=None):
        base = self.base_model()
        if not base.get("word_next") or not base.get("word_starts"):
            return None
        if length is None:
            length = random.randint(18, 70)
        length = max(8, min(int(length), self.generation_max))

        starts = [pair for pair in base["word_starts"] if len(pair) == 2]
        if not starts:
            return None

        # Select starting phrases from learned human text that shares a topic with the message.
        # This makes the seed affect generation rather than merely being passed through unused.
        hint_words = self.hint_words(hints)
        if seed or hint_words:
            seed_tokens = {
                token for token in self.word_tokens(seed)
                if len(token) >= 2 and not nuunre.fullmatch(token)
            } | hint_words
            indexed_candidates = {}
            topic_index = base.get("topic_index", {})
            for token in seed_tokens:
                for entry in topic_index.get(token, ()):
                    indexed_candidates[entry] = None
            topic_starts = []
            for start_pair, candidate_tokens in indexed_candidates:
                overlap = len(seed_tokens & candidate_tokens) / max(1, len(seed_tokens | candidate_tokens))
                if overlap >= 0.16:
                    topic_starts.append((overlap, start_pair))
            if topic_starts:
                best_score = max(item[0] for item in topic_starts)
                near_best = [pair for score, pair in topic_starts if score >= best_score * 0.75]
                words = list(random.choice(near_best))
            else:
                words = list(random.choice(starts))
        else:
            words = list(random.choice(starts))
        natural = False
        max_tokens = max(12, min(80, length * 2))

        for _ in range(max_tokens):
            current = "".join(words)
            if len(current) >= max(12, int(length * 0.45)) and current[-1:] in "。！？!?…~〜":
                natural = True
                break
            candidates = base["word_next"].get(self.word_key(words[-1:]))
            if not candidates:
                break
            if hint_words:
                # 学習済みの情報に出てくる言葉を選ばれやすくする。文そのものは使わない。
                candidates = self.bias(candidates, hint_words, 6)
            nxt = self.weighted_choice(candidates)
            if not nxt:
                break
            words.append(nxt)
            if len("".join(words)) >= length and nxt and re.fullmatch(r"[。！？!?…~〜]+", nxt):
                natural = True
                break
            if len("".join(words)) >= self.generation_max:
                break

        output = "".join(words).replace("\\n", "").replace("\\r", "")
        output = re.sub(r"\s+", " ", output).strip()
        if not natural or not output or output[-1] not in "。！？!?…~〜":
            return None
        if not nuunre.search(output):
            return None
        return output[:self.generation_max]

    def finish(self, output, natural):
        if not output or not natural:
            return None
        if output[-1] not in "。！？!?…~〜":
            return None
        if not nuunre.search(output):
            return None
        return output

    def learn_conversation(self, message, response):
        message = self.clean(message)
        response = self.clean(response)

        if message:
            self.learn(message, 1, talk=True)
        if response:
            self.learn(response, 4)

    def save_if_needed(self, every=30):
        if self.dirty >= every:
            self.save()


nuun = NuunGakushu()
nuun.yomu_settings()    # 返信確率などの設定だけ読む(マルコフ用の基礎データ構築は不要になった)


def yobareta(m):
    if bot.user in m.mentions:
        return True
    ref = m.reference.resolved if m.reference else None
    return isinstance(ref, discord.Message) and ref.author == bot.user


class Bot(commands.Bot):
    async def setup_hook(self):
        await self.tree.sync()
        # 意味検索(ruri-v3)の読み込みは裏で行う。完了するまでは語の重なりで代用する。
        self.vec_task = asyncio.create_task(vec_index.start(shared_memory))
        asyncio.create_task(asyncio.to_thread(get_tokenizer))    # 形態素解析の辞書を先に読み込んでおく

    async def close(self):
        shared_memory.save()
        await asyncio.to_thread(vec_index.save)
        await super().close()


intents = discord.Intents.default()
intents.message_content = True
bot = Bot(command_prefix="!", intents=intents)


nuunchang = app_commands.Group(name="ぬーんちゃん", description="あなたのぬーんちゃん", guild_only=True)
bot.tree.add_command(nuunchang)


@nuunchang.command(name="呼び方", description="ぬーんちゃんの呼び方を設定・確認")
@app_commands.rename(namae="名前")
@app_commands.describe(namae="呼び方。省略すると現在の呼び方を表示")
async def nuunchanyobikata_cmd(yobi, namae: Optional[str] = None):
    u = nuunchanuser(yobi.user.id)
    if namae is None:
        return await yobi.response.send_message(f"あなたのぬーんちゃんの呼び方は「{u['yobikata']}」ぬーん", ephemeral=True)
    namae = namae.strip()[:30]
    if not namae:
        return await yobi.response.send_message("呼び方が空っぽぬーん", ephemeral=True)
    u["yobikata"] = namae
    nuunchankaku()
    await yobi.response.send_message(f"これから「{namae}」と呼ばれたら出てくるぬーん！", ephemeral=True)


@bot.tree.command(name="ぬーんランキング", description="ぬーんランキングを見る")
async def nuunranking(yobi):
    ranking_text = await nuunchanranktext(yobi.user.id, yobi.guild)
    try:
        await yobi.response.send_message(ranking_text)
    except discord.Forbidden:
        if not yobi.response.is_done():
            await yobi.response.send_message("このチャンネルではランキングを表示できないぬーん。Botの権限を確認してほしいぬーん。", ephemeral=True)

globalchat = app_commands.Group(name="グローバルチャット", description="サーバー間でチャットを共有", guild_only=True)
bot.tree.add_command(globalchat)


@globalchat.command(name="参加", description="このチャンネルをグローバルチャットに参加させる")
async def globalchat_join(yobi):
    if not yobi.permissions.manage_guild:
        return await yobi.response.send_message("管理権限が必要ぬーん", ephemeral=True)
    globalchat_add(yobi.channel.id)
    await yobi.response.send_message("このチャンネルをグローバルチャットに参加させたぬーん")


@globalchat.command(name="退出", description="このチャンネルをグローバルチャットから退出させる")
async def globalchat_leave(yobi):
    if not yobi.permissions.manage_guild:
        return await yobi.response.send_message("管理権限が必要ぬーん", ephemeral=True)
    if globalchat_remove(yobi.channel.id):
        await yobi.response.send_message("このチャンネルをグローバルチャットから退出させたぬーん")
    else:
        await yobi.response.send_message("このチャンネルはグローバルチャットに入っていないぬーん", ephemeral=True)


@globalchat.command(name="確認", description="グローバルチャットの登録状況を確認")
async def globalchat_check(yobi):
    groups = []
    for cid in globalchat_channels():
        ch = bot.get_channel(int(cid))
        if ch is None:
            continue
        guild = getattr(ch, "guild", None)
        if guild is None:
            continue
        name = getattr(ch, "name", "不明")
        groups.append(f"{guild.name}\n→ {name}")
    text = "\n\n".join(groups) if groups else "なし"
    await yobi.response.send_message("グローバルチャット参加チャンネル\n\n" + text, ephemeral=True)


banpu = app_commands.Group(name="バンプ", description="Bump通知の設定", guild_only=True)
hannou = app_commands.Group(name="反応", description="このチャンネルの自動反応", guild_only=True)
for g in (banpu, hannou):
    bot.tree.add_command(g)


@banpu.command(name="設定", description="Bump通知を設定")
@app_commands.rename(joutai="状態")
async def banpukae(yobi, joutai: Kiri):
    if not yobi.permissions.manage_guild:
        return await yobi.response.send_message("管理権限が必要ぬーん", ephemeral=True)
    saba(yobi)["banpu"] = joutai == "オン"
    kaku()
    await yobi.response.send_message(f"バンプ通知を{joutai}にしたぬーん")


@banpu.command(name="確認", description="Bump通知を確認")
async def banpumi(yobi):
    await yobi.response.send_message(f"バンプ通知は{onofu(saba(yobi)['banpu'])}ぬーん")


@hannou.command(name="設定", description="自動反応を設定")
@app_commands.rename(joutai="状態")
async def hannoukae(yobi, joutai: Kiri):
    if not yobi.permissions.manage_guild:
        return await yobi.response.send_message("管理権限が必要ぬーん", ephemeral=True)
    saba(yobi)["hannou"][str(yobi.channel.id)] = joutai == "オン"
    kaku()
    await yobi.response.send_message(f"このチャンネルの自動反応を{joutai}にしたぬーん")


@hannou.command(name="確認", description="自動反応の状態を確認")
async def hannoumi(yobi):
    await yobi.response.send_message(
        f"このチャンネルの自動反応は{onofu(han(yobi))}、{ritsusetsu(yobi)}ぬーん"
    )


def kengen(yobi):
    return yobi.permissions.manage_guild


NASHI = "管理権限が必要ぬーん"
MEMO = "\n(自動反応がオフなので、オンにするまで反応しないぬーん)"


@hannou.command(name="サーバー確率", description="サーバー全体の返信確率を設定(0〜100)")
@app_commands.rename(parasento="パーセント")
@app_commands.describe(parasento="返信確率(パーセント)。省略すると今の値を表示")
async def hannousaba(yobi, parasento: Optional[app_commands.Range[int, 0, 100]] = None):
    if parasento is None:
        v = ritsusaba(yobi)
        return await yobi.response.send_message(
            f"サーバー全体の返信確率は{'未設定' if v is None else f'{v}パーセント'}ぬーん"
        )
    if not kengen(yobi):
        return await yobi.response.send_message(NASHI, ephemeral=True)
    saba(yobi)["ritsu_saba"] = int(parasento)
    kaku()
    await yobi.response.send_message(f"サーバー全体の返信確率を{parasento}パーセントにしたぬーん")


@hannou.command(name="サーバー確率リセット", description="サーバー全体の返信確率を初期値に戻す")
async def hannousabakeshi(yobi):
    if not kengen(yobi):
        return await yobi.response.send_message(NASHI, ephemeral=True)
    saba(yobi).pop("ritsu_saba", None)
    kaku()
    await yobi.response.send_message("サーバー全体の返信確率を初期値に戻したぬーん")


@hannou.command(name="チャンネル確率", description="このチャンネルの返信確率を設定(0〜100)")
@app_commands.rename(parasento="パーセント")
@app_commands.describe(parasento="返信確率(パーセント)。省略すると今の値を表示")
async def hannouchan(yobi, parasento: Optional[app_commands.Range[int, 0, 100]] = None):
    if parasento is None:
        v = ritsu(yobi)
        return await yobi.response.send_message(
            f"このチャンネルの返信確率は{'未設定' if v is None else f'{v}パーセント'}ぬーん"
        )
    if not kengen(yobi):
        return await yobi.response.send_message(NASHI, ephemeral=True)
    saba(yobi).setdefault("ritsu", {})[str(yobi.channel.id)] = int(parasento)
    kaku()
    await yobi.response.send_message(
        f"このチャンネルの返信確率を{parasento}パーセントにしたぬーん{'' if han(yobi) else MEMO}"
    )


@hannou.command(name="チャンネル確率リセット", description="このチャンネルの返信確率を初期値に戻す")
async def hannouchankeshi(yobi):
    if not kengen(yobi):
        return await yobi.response.send_message(NASHI, ephemeral=True)
    saba(yobi).setdefault("ritsu", {}).pop(str(yobi.channel.id), None)
    kaku()
    await yobi.response.send_message("このチャンネルの返信確率を初期値に戻したぬーん")


@hannou.command(name="個人確率", description="あなたへの返信確率を設定(0〜100)")
@app_commands.rename(parasento="パーセント")
@app_commands.describe(parasento="返信確率(パーセント)。省略すると今の値を表示")
async def hannoukojin(yobi, parasento: Optional[app_commands.Range[int, 0, 100]] = None):
    if parasento is None:
        v = ritsukojin(yobi)
        return await yobi.response.send_message(
            f"あなたへの返信確率は{'未設定' if v is None else f'{v}パーセント'}ぬーん", ephemeral=True
        )
    saba(yobi).setdefault("ritsu_kojin", {})[str(yobi.user.id)] = int(parasento)
    kaku()
    await yobi.response.send_message(f"あなたへの返信確率を{parasento}パーセントにしたぬーん", ephemeral=True)


@hannou.command(name="個人確率リセット", description="あなたへの返信確率の設定を消す")
async def hannoukojinkeshi(yobi):
    saba(yobi).setdefault("ritsu_kojin", {}).pop(str(yobi.user.id), None)
    kaku()
    await yobi.response.send_message("あなたへの返信確率の設定を消したぬーん", ephemeral=True)


@hannou.command(name="一覧", description="反応一覧")
async def hannouichi(yobi):
    if not kotoba["hannou"]:
        return await yobi.response.send_message("反応は登録されていないぬーん")
    await yobi.response.send_message("\n".join(
        f"{n} {'キーワード' if r.get('kiwado') else '常時'} "
        f"{r.get('ritsu', 0)}パーセント {r.get('bun') or '画像のみ'}"
        for n, r in enumerate(kotoba["hannou"], 1)
    ))


@bot.tree.command(name="ぬーんガチャ", description="ぬーんガチャを引く")
async def gacha(yobi):
    await yobi.response.send_message(**hiku())


@bot.tree.command(name="設定確認", description="サーバー設定を確認")
@app_commands.guild_only()
async def miru(yobi):
    await yobi.response.send_message(
        f"バンプ通知は{onofu(saba(yobi)['banpu'])}ぬーん\n"
        f"このチャンネルの自動反応は{onofu(han(yobi))}、{ritsusetsu(yobi)}ぬーん"
    )


chishiki = app_commands.Group(name="知識", description="ぬーんに知識を覚えさせる", guild_only=True)
bot.tree.add_command(chishiki)


def chishikiomoji(text, n=90):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n - 1] + "…"


@chishiki.command(name="教える", description="ぬーんに知識を1つ教える(全サーバー共通)")
@app_commands.rename(shugo="主語", naiyou="内容")
@app_commands.describe(shugo="何についての知識か(例: ぬーん)", naiyou="内容(例: Discordのbotだ)")
async def chishikioshieru(yobi, shugo: app_commands.Range[str, 1, 20], naiyou: app_commands.Range[str, 1, 140]):
    shugo, naiyou = shugo.strip(), naiyou.strip()
    if re.search(r"https?://|<@|<#|@everyone|@here", shugo + naiyou):
        return await yobi.response.send_message("URLやメンションは覚えられないぬーん", ephemeral=True)
    text = naiyou if SharedMemory.normalize(shugo) in SharedMemory.normalize(naiyou) else f"{shugo}は{naiyou}"
    r = shared_memory.chishiki_oboeru(text, yobi.user.id, yobi.guild.id, subject=shugo, explicit=True)
    if r is None:
        return await yobi.response.send_message("覚えられなかったぬーん", ephemeral=True)
    shared_memory.save()
    await yobi.response.send_message(f"覚えたぬーん({r})：{text}")


@chishiki.command(name="忘れる", description="キーワードを含む知識を消す(管理権限なら自分のサーバーで覚えたもの、なければ自分だけが教えたもの)")
@app_commands.rename(kiwado="キーワード")
async def chishikiwasureru(yobi, kiwado: app_commands.Range[str, 1, 40]):
    n = shared_memory.chishiki_wasureru(kiwado, yobi.user.id, yobi.guild.id, kengen(yobi))
    if not n:
        return await yobi.response.send_message("消せる知識がなかったぬーん", ephemeral=True)
    shared_memory.save()
    await yobi.response.send_message(f"「{kiwado}」を含む知識を{n}件忘れたぬーん")


@chishiki.command(name="一覧", description="覚えている知識を見る")
@app_commands.rename(kiwado="キーワード")
@app_commands.describe(kiwado="省略すると全体の状況を表示")
async def chishikiichiran(yobi, kiwado: Optional[str] = None):
    if kiwado:
        hits = shared_memory.chishiki_kensaku(kiwado)
        if not hits:
            return await yobi.response.send_message("その知識はまだないぬーん", ephemeral=True)
        lines = [f"{'📌' if f['x'] else '・'}{chishikiomoji(f['t'])}" for f in hits[-12:]]
        return await yobi.response.send_message(
            f"{len(hits)}件ヒット(新しい順に最大12件、📌=教えてもらったもの)\n" + "\n".join(lines), ephemeral=True
        )
    total = len(shared_memory.facts)
    if not total:
        return await yobi.response.send_message("まだ何も覚えていないぬーん", ephemeral=True)
    explicit = sum(1 for f in shared_memory.facts if f["x"])
    top = sorted(shared_memory.chishiki_idx.items(), key=lambda kv: len(kv[1]), reverse=True)[:30]
    await yobi.response.send_message(
        f"全サーバー共通で{total}件(うち教えてもらったもの{explicit}件)覚えているぬーん\n"
        + "、".join(f"{k}({len(v)})" for k, v in top),
        ephemeral=True,
    )


@chishiki.command(name="検索", description="意味検索の結果と類似度を確認する(しきい値の調整用)")
@app_commands.rename(bun="文章")
async def chishikikensaku(yobi, bun: app_commands.Range[str, 1, 200]):
    head = f"意味検索: {vec_index.state}({len(vec_index.items)}件)"
    found = vec_index.search(bun)
    if found is None:
        return await yobi.response.send_message(head + "\n今は語の重なりで探しているぬーん", ephemeral=True)
    lines = []
    for kind, label, limit in (("f", "知識", vec_fact_min), ("t", "会話", vec_turn_min)):
        for score, it in found[kind][:3]:
            body = it["t"] if kind == "f" else f"{it['t']} → {it.get('a', '')}"
            lines.append(f"{'✅' if score >= limit else '・'}{score:.2f} [{label}] {chishikiomoji(body, 70)}")
    if not lines:
        lines.append("近いものはまだないぬーん")
    await yobi.response.send_message(
        head + f"\n✅=ヒントに使う(知識{vec_fact_min}以上 / 会話{vec_turn_min}以上)\n" + "\n".join(lines), ephemeral=True
    )


@chishiki.command(name="履歴学習", description="チャンネルの過去発言から知識を学習する(管理権限が必要)")
@app_commands.rename(channel="チャンネル", kensuu="件数")
@app_commands.describe(channel="省略すると今のチャンネル", kensuu="さかのぼる発言数(10〜1000、省略は200)")
async def chishikirireki(yobi, channel: Optional[discord.TextChannel] = None, kensuu: app_commands.Range[int, 10, 1000] = 200):
    if not kengen(yobi):
        return await yobi.response.send_message(NASHI, ephemeral=True)
    channel = channel or yobi.channel
    me = yobi.guild.me
    if not isinstance(channel, discord.TextChannel) or not channel.permissions_for(me).read_message_history:
        return await yobi.response.send_message("そのチャンネルの履歴を読めないぬーん", ephemeral=True)
    if yobi.guild.id in chishiki_crawling:
        return await yobi.response.send_message("このサーバーはもう履歴学習中ぬーん", ephemeral=True)
    chishiki_crawling.add(yobi.guild.id)
    await yobi.response.defer(thinking=True)
    new = upd = seen = 0
    try:
        async for m in channel.history(limit=kensuu):
            if m.author.bot or not m.content or m.content.startswith(("/", "!")):
                continue
            seen += 1
            a, b = shared_memory.chishiki_manabu(m.content, m.author.id, yobi.guild.id)
            new += a
            upd += b
    except discord.Forbidden:
        return await yobi.followup.send("履歴を読む権限が途中でなくなったぬーん")
    finally:
        chishiki_crawling.discard(yobi.guild.id)
        shared_memory.save()
    await yobi.followup.send(f"{channel.mention} の{seen}件を読んで、知識を{new}件覚えて、{upd}件は重複だったぬーん")


@bot.event
async def on_raw_reaction_add(payload):
    await globalchat_reaction(payload, True)


@bot.event
async def on_raw_reaction_remove(payload):
    await globalchat_reaction(payload, False)


@bot.event
async def on_message_edit(before, after):
    await globalchat_edit(after)


@bot.event
async def on_message_delete(message):
    await globalchat_delete(message)


@bot.event
async def on_ready():
    print(f"ログインしました {bot.user}")
    print(f"参加サーバー: {len(bot.guilds)}")
    for guild in sorted(bot.guilds, key=lambda g: g.name.casefold()):
        print(guild.name)


@bot.event
async def on_message(hatsu):
    if not hatsu.guild:
        return

    if hatsu.author.bot and hatsu.author.id != bot.user.id:
        if hatsu.author.id == banpuid and saba(hatsu)["banpu"]:
            await okuru(hatsu.channel, kotoba["banpu"])
        return

    await globalchat_relay(hatsu)

    if hatsu.author.bot:
        return

    context_for_recall = recent_human_messages.get(hatsu.channel.id)

    if hatsu.content and not hatsu.content.startswith(("/", "!")):
        learned_text = nuun.clean(hatsu.content)
        if len(learned_text) >= 2:
            shared_memory.learn(learned_text)
        shared_memory.chishiki_manabu(hatsu.content, hatsu.author.id, hatsu.guild.id)
        if len(learned_text) >= 4:

            referenced = hatsu.reference.resolved if hatsu.reference else None
            if (
                isinstance(referenced, discord.Message)
                and not referenced.author.bot
                and referenced.author.id != hatsu.author.id
                and referenced.content
            ):
                shared_memory.learn_turn(referenced.content, learned_text)
            else:
                previous = recent_human_messages.get(hatsu.channel.id)
                now_for_learning = time.monotonic()
                if (
                    previous
                    and previous[0] != hatsu.author.id
                    and now_for_learning - previous[2] <= 90
                ):
                    shared_memory.learn_turn(previous[1], learned_text)

            recent_human_messages[hatsu.channel.id] = (
                hatsu.author.id,
                learned_text,
                time.monotonic(),
            )

    if shared_memory.dirty:
        shared_memory.save_async()    # 学習するごとにその場で書き換える(裏のスレッド。返信は待たせない)

    now = time.monotonic()
    q = renpo[hatsu.author.id]
    while q and now - q[0] >= renpo_window:
        q.popleft()
    if len(q) >= renpo_limit:
        nuun.save_if_needed()
        if shared_memory.dirty:
            shared_memory.save_async()
        await bot.process_commands(hatsu)
        return
    q.append(now)

    nuun_count = len(nuunre.findall(hatsu.content))
    if nuun_count:
        nuunchanrank(hatsu.author.id, hatsu.guild.id, nuun_count)
        nuunchankaku()

                                 
    context_text = context_for_recall[1] if context_for_recall and time.monotonic() - context_for_recall[2] <= 120 else None
    message_text = hatsu.content.strip()
    is_question = (
        any(mark in message_text for mark in ("?", "？"))
        or SharedMemory.chishiki_wadai(message_text)[1]
        or re.search(r"(?:は|って)[?？]$", message_text) is not None
        or (context_text is not None and len(SharedMemory.normalize(message_text)) <= 8)
    )
    # 学習した知識や過去の会話は、保存された文をそのまま返すためには使わない。
    # 返信は必ず通常の生成(下の hentou)で作り、ここで探した情報は「話題と語彙を寄せるヒント」として渡すだけにする。
    # チャンネルの反応オン/オフ・返信確率・呼ばれたかどうかの判定も、通常の生成と同じものを通す。
    generation_seed = message_text

    pet_response = nuunchanart(hatsu.content, hatsu.author.id)
    if pet_response:
        await hatsu.channel.send(pet_response)
        nuunchankaku()
        nuun.save_if_needed()
        await bot.process_commands(hatsu)
        return

    if hatsu.content.strip() == "ぬーんランキング":
        try:
            await hatsu.channel.send(await nuunchanranktext(hatsu.author.id, hatsu.guild))
        except discord.Forbidden:
            print(f"ランキングを送信できません: channel_id={hatsu.channel.id}。Botにメッセージ送信権限がありません。")
        except discord.HTTPException as error:
            print(f"ランキング送信エラー: {error}")
        nuunchankaku()
        nuun.save_if_needed()
        await bot.process_commands(hatsu)
        return

                        
    if gachago in hatsu.content:
        await hatsu.channel.send(**hiku())
        await bot.process_commands(hatsu)
        return

                                                      
    fixed_reacted = False

    if han(hatsu) and random.random() < koteiritsu:
        for r in kotoba.get("hannou", []):
            keywords = r.get("kiwado", r.get("kotoba", []))
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [str(k).strip() for k in keywords if str(k).strip()]

            if keywords and hatsu.content.strip() not in keywords:
                continue

            try:
                ritsu = float(r.get("ritsu", 0))
            except (TypeError, ValueError):
                ritsu = 0

            if random.random() * 100 >= ritsu:
                continue

            await okuru(hatsu.channel, r)
            fixed_reacted = True
            break

    if fixed_reacted:
        nuun.save_if_needed()
        await bot.process_commands(hatsu)
        return

    if han(hatsu):
        try:
            generation_chance = float(nuun.settings.get("generation_chance", 0.22))
        except (TypeError, ValueError):
            generation_chance = 0.22

                                              
        kimaru, moto = ritsukimaru(hatsu)
        if moto != "初期値":
            generation_chance = kimaru / 100

                              
        if yobareta(hatsu):
            generation_chance = 1.0

        if generation_chance > 0 and random.random() < generation_chance:
            reply = await kotaeru(message_text, context_text, is_question, hatsu.channel.id)
            if not reply and yobareta(hatsu):
                reply = random.choice(["ぬーん？", "ぬーん", "呼んだぬーん？"])
            if reply:
                await hatsu.channel.send(reply, allowed_mentions=discord.AllowedMentions.none())
                # Bot自身の返信は学習に使わない。

                                               
    nuun.save_if_needed()
    if shared_memory.dirty:
        shared_memory.save_async()
    await bot.process_commands(hatsu)


if __name__ == "__main__":
    if tokun == "ここにBotトークンを入れてください":
        raise SystemExit("mein.py の tokun にBotトークンを設定してください")
    bot.run(tokun)
