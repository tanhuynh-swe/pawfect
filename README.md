# Pawfect Love Animals — build, automate, monetize

A complete setup for `@PawfectLoveAnimals`: a plan for the channel, and a working
Python pipeline that researches a topic, builds the script, narrates it, finds
footage, renders a finished 1080p video, makes a thumbnail, and schedules the
upload to YouTube.

Read the first section before you run anything. It is the part that decides
whether this channel makes money or gets demonetized.

---

## 1. The one rule that changes the whole design

You asked for a channel that runs without you touching it. I built as much of
that as can safely be built, but I have to be straight with you about where the
limit is, because getting this wrong wastes months.

YouTube's **inauthentic content policy** (updated July 2025, enforced hard
through 2026) blocks monetization for content that is *mass-produced or
repetitive* — videos that look templated, with minimal variation, made at a
scale that implies no human judgment. YouTube removed thousands of fully
automated channels under this policy. It is not aimed at AI specifically: an
AI-assisted channel with a real editorial point of view is fine, and a
human-made channel of copy-paste videos is not.

Where channels actually land:

| Risk | What it looks like |
|---|---|
| **High** | Scraped scripts, synthetic voice, generic stock footage, daily uploads, nobody reviews anything |
| **Medium** | Heavy AI production, identical structure every video, re-narrating other people's videos |
| **Low** | AI drafts a script *you revise*, AI voice, AI editing — inside work that has a genuine angle |

So the pipeline is built for the **low-risk** column. It automates everything
mechanical — research, narration, footage, editing, rendering, thumbnails,
uploading, scheduling — and stops at one gate: you watch the video and approve
it before it publishes. That gate costs you about ten minutes per video. It is
the difference between a channel that earns and a channel that gets terminated.

You *can* turn the gate off (`safety.require_human_review: false`). I would not,
at least not until the channel is monetized and you know what your videos look
like.

Two smaller facts worth knowing:

- **AI disclosure.** You only need YouTube's "altered or synthetic content"
  label for *realistic* synthetic media — making a real person appear to say
  something, or generating realistic footage of events that did not happen. An
  AI voice narrating licensed stock footage does **not** need the label. The
  pipeline leaves `contains_synthetic_media: false` and you flip it per video if
  you ever generate realistic scenes.
- **The "30% commentary rule" is fake.** So is the "20% script variation
  threshold." Those numbers circulate on AI-tool blogs and appear in no YouTube
  policy. Ignore them.

---

## 2. What you need to reach, to get paid

Two tiers. You are aiming at the second one.

**Tier 1 — fan funding** (channel memberships, Super Thanks): 500 subscribers,
3 public uploads in the last 90 days, and 3,000 public watch hours in 12 months.

**Tier 2 — ad revenue** (the real money): 1,000 subscribers and 4,000 public
watch hours in 12 months. You keep 55% of ad revenue on long-form.

One thing to plan around: for creators **applying from 1 February 2027**, the
long-form requirement rises to **8,000 watch hours**. If you can get to 4,000
hours and apply before that date, you are judged on the old bar. That is a real
reason to start publishing now rather than in six months.

The arithmetic on 4,000 hours, for 7-minute videos with a typical 40% average
view duration: each view is worth roughly 2.8 minutes, so you need about 86,000
views. At 3 videos a week that is very achievable inside a year in the pet
niche *if the videos are good* — and impossible at any volume if they are not.
Volume is not the lever. Retention is.

Pet content also pays above average. English-language pet care sits in a
comfortable CPM range because pet food, insurance and vet-service advertisers
compete for it, and your audience skews toward people who spend money on their
animals. Affiliate links to pet products will likely out-earn ad revenue in
your first year — put them in descriptions from video one, disclosed.

---

## 3. The channel plan

**Positioning.** "Pawfect Love Animals" is a warm name, so the content should be
warm but *useful*. The angle I put in `config.yaml` is:

> Practical, evidence-based pet care explained calmly for first-time owners,
> with the vet consensus stated plainly and the common myths named and corrected.

That angle is the most important line in the whole project. It is what makes
your video different from the five other AI pet channels covering the same
topic, and "could a viewer swap your channel for five competitors and notice
nothing?" is precisely the test YouTube's policy applies. Rewrite it in your own
words once you see which videos your audience responds to.

**Format.** Long-form only, as you chose: 6–9 minutes. Long enough for real ad
revenue and watch hours, short enough that retention holds. Every video answers
one specific question a worried owner typed into search.

**Cadence.** Three a week — Tuesday, Thursday, Saturday at 17:00 your time
(already set in `config.yaml`). Do not go daily. Daily output is the single
strongest signal of a content farm, and it will not double your growth.

**Topics.** Fifteen are seeded in `topics/seed_topics.yaml`. They are all
specific search questions with a stated angle, because "dog facts" gets no
traffic and "why does my dog eat grass" gets a lot. When you run out, take
topics from your own comments — that is also the best possible evidence to
YouTube that a human runs this channel.

**First 90 days, realistically.** Videos 1–10 will get very few views and that
is normal; you are building a library and learning what your thumbnails should
look like. Somewhere between video 15 and 40, one video usually outperforms the
rest by 10x. Then you make five more videos on that exact subject. That is the
whole growth strategy, and no amount of automation replaces it.

---

## 4. Cost

Everything is free. Total: **$0/month**.

| Job | Tool | Cost |
|---|---|---|
| Search demand | Google's public suggest endpoint | free, no key |
| Facts | Wikipedia API | free, no key |
| Script | Claude app (paste the prompt) | free with your plan |
| Voice | Edge TTS (English), VieNeu-TTS (Vietnamese), Piper offline | free, MIT / Apache 2.0 |
| Footage | Pexels + Pixabay APIs | free key, commercial use, no attribution |
| Editing | ffmpeg | free |
| Thumbnail | Pillow | free |
| Upload | YouTube Data API v3 | free |

The one optional paid upgrade: set `script.mode: api` with an Anthropic API key
and scripts write themselves for roughly a few cents each. You'd still review
them. Worth it once you are making three videos a week and the copy-paste step
starts to annoy you.

---

## 5. Setup on your Mac mini

The pipeline runs on your own machine, not in the cloud. About 30 minutes, once.

### 5.1 Install

```bash
brew install ffmpeg python@3.11
cd ~/pawfect
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Free API keys

```bash
cp .env.example .env
```

Get both keys (no credit card for either) and paste them into `.env`:

- Pexels: <https://www.pexels.com/api/>
- Pixabay: <https://pixabay.com/api/docs/>

### 5.3 YouTube upload access

1. Open <https://console.cloud.google.com/> and create a project.
2. Enable **YouTube Data API v3**.
3. OAuth consent screen: External, add `tanhuynhswe@gmail.com` as a test user.
4. Credentials → Create credentials → OAuth client ID → **Desktop app**.
5. Download the JSON, save it as `client_secret.json` in the project folder.

**Expect this on your first upload:** videos uploaded through the API from a
project that has not passed YouTube's compliance audit are **locked to private**.
`publishAt` will not make them public. The pipeline works fine — you just flip
each video to public yourself in YouTube Studio until the audit clears. Apply
here once you have a few videos up:
<https://support.google.com/youtube/contact/yt_api_form>

Also: custom thumbnails require a **verified phone number** on the channel.
Do that now in YouTube Studio, it takes a minute.

### 5.4 Background music (optional but recommended)

Drop one or two royalty-free tracks into `assets/music/`. The YouTube Audio
Library (Studio → Audio library) is free and safe. The pipeline ducks the music
under the narration automatically. With no track, you get clean narration only.

---

## 6. Daily use

```bash
python run.py make
```

This picks the best unused topic, scores it against real search demand, pulls
grounding facts, and writes `workspace/<slug>/prompt.txt`.

Open that file, paste it into the Claude app, and save the JSON reply as
`workspace/<slug>/script.json`. **Read the script while you paste it.** Fix
anything that sounds wrong — that is your editorial pass, and it is what the
whole channel's compliance rests on.

Then:

```bash
python run.py build <slug>      # narration, footage, render, thumbnail — ~5 min
open workspace/<slug>/final.mp4 # watch it
python run.py approve <slug>
python run.py publish <slug>    # uploads, scheduled to your next slot
```

Other commands:

```bash
python run.py status                        # what's in the pipeline
python run.py topic --title "your idea"     # force a specific topic
python run.py publish <slug> --at 2026-10-02T10:00:00Z
python run.py auto                          # the whole thing, unattended
python run.py resume                        # clear a self-imposed pause
```

---

## 6b. Fully unattended mode

`run.py auto` does the whole thing with no input from you: picks a topic, writes
the script through the Anthropic API, renders, checks its own work, uploads and
publishes, then emails you what it did.

### What you need for it

1. `ANTHROPIC_API_KEY` in `.env` — the copy-paste script step cannot run
   unattended, so auto mode requires `script.mode: api` (already set).
2. A Gmail **App Password** in `.env` for the digest. Not your normal password:
   create one at <https://myaccount.google.com/apppasswords> (2-Step
   Verification must be on first).

### The audit blocker — read this one

Until your Google Cloud project passes YouTube's compliance audit, **every video
uploaded through the API is forced to private**, whatever the code asks for.
That is YouTube's rule and no setting changes it.

So auto mode works in two phases:

- **Before the audit:** everything runs automatically, videos land on your
  channel as private, and the digest email tells you to flip each one public in
  Studio. That is about one minute per video.
- **After the audit:** identical code, videos go live on their own. Nothing to
  change.

Apply once you have a few videos up:
<https://support.google.com/youtube/contact/yt_api_form>

### The automated gate

Nobody is watching before publish, so the robot reviews itself. A video is
**held, not published**, if any of these are true:

| Check | Why it blocks |
|---|---|
| Claude's review returns HOLD | Invented statistics, unsafe advice, or claims stated more confidently than the evidence supports |
| Script is >30% similar to one of your published scripts | This is precisely the "mass-produced or repetitive" pattern YouTube demonetizes for. It is the most valuable check here |
| The review could not run at all | A failed check never silently becomes a pass |
| Runtime is far under target | Usually means the script generation went wrong |

Filler phrases and an over-long title are reported but do not block.

If three runs in a row are held, auto mode **pauses itself** and emails you —
a run of holds means something upstream broke, and publishing through it would
make things worse. Clear it with `python run.py resume`.

Held videos stay in `workspace/<slug>/` with a `qa.json` explaining exactly what
failed. Fix the script and run `python run.py build <slug> && python run.py
publish <slug>`.

### Install the schedule

```bash
./install_schedule.sh
```

This installs a `launchd` agent that runs `run.py auto` at 08:00 on Tuesday,
Thursday and Saturday. I used `launchd` rather than cron because cron jobs on
macOS die quietly when the Mac sleeps, while launchd catches up a missed run
on wake.

```bash
launchctl kickstart -p gui/$(id -u)/com.pawfect.autopublish   # run it right now
tail -f logs/auto.log                                          # watch it work
./install_schedule.sh remove                                   # uninstall
```

Stop the Mac mini sleeping through its schedule:

```bash
sudo pmset -a sleep 0 disksleep 0
```

### What a run costs

Script plus automated review is roughly 3–6 cents of Anthropic API per video,
so about 50 cents a month at three videos a week. Everything else stays free.

### Before you leave it alone

Run it manually two or three times first and watch the output. The gate catches
bad facts and repetition; it cannot catch a voice that mispronounces a breed
name, or footage of a cat in a video about dogs. Once you have seen a few videos
come out right, the schedule is safe to trust.

### Manual mode still works

`run.py make` and the approve/publish commands are unchanged, and
`safety.require_human_review: true` still protects that path. You can use auto
mode on weekdays and hand-make a video when you have a better idea.

---

## 7. How the pipeline works

```
topics/seed_topics.yaml
        │
        ▼
  research.py ──── Google suggest (what people search)
        │     └─── Wikipedia (facts to check against)
        ▼
   brief.json
        │
        ▼
   script.py ───── prompt.txt → you + Claude → script.json
        │                        ▲
        ▼                        └── YOUR EDITORIAL PASS
    scenes[]
        │
        ├──► voice.py ──── Edge / VieNeu / Piper TTS → one wav per scene, measured
        │
        ├──► visuals.py ── Pexels / Pixabay → one clip per shot, searched on
        │                  that shot's own query, never reused in a video
        │
        └──► render.py ─── one shot per 6.5s, burned captions, ducked music,
                           loudness-normalized to −16 LUFS → final.mp4
                                     │
                              thumbnail.py
                                     │
                              [ APPROVAL GATE ]
                                     │
                               upload.py → YouTube, scheduled
```

A few decisions in there that matter:

**One visual per 6.5 seconds, not one per scene.** A 40-second scene gets six
different clips. This is the single biggest retention lever in the file — long
static shots are where viewers leave.

**Captions are burned in.** A large share of pet content is watched with sound
off at some point. Unburned captions lose those viewers.

**Audio is normalized to −16 LUFS** with the music side-chain ducked under the
voice. YouTube normalizes loudness anyway; matching its target means your video
does not sound quieter than everything else in the feed.

**Topics are never reused.** `workspace/used_topics.json` tracks what has
published.

---

## 8. Making money, in order

1. **Affiliate links from video one.** Amazon Associates, Chewy. Link the
   specific products a video mentions. Disclose it in the description. This
   earns before monetization does.
2. **YPP at 1,000 subs / 4,000 hours.** Apply the moment you qualify, and
   ideally before 1 February 2027 while the bar is 4,000 hours.
3. **Channel memberships** once you have a regular audience.
4. **Sponsorships.** Pet brands approach niche channels at surprisingly small
   subscriber counts — around 10k — if the audience is tightly defined. Your
   angle is what makes it tightly defined.

---

## 9. Things that will go wrong, and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| Uploads stay private forever | API project not audited | Flip to public in Studio; apply for the audit |
| `thumbnail rejected (403)` | Channel phone not verified | Verify in YouTube Studio |
| "no stock match", colored cards | Pexels/Pixabay key missing or query too abstract | Check `.env`; make each `visual_queries` entry concrete and filmable |
| Shots don't match what is being said | Scene has one query for a long scene | Give the scene one entry in `visual_queries` per ~6.5s of its narration, in spoken order |
| Video is 3 min, not 7 | Script too short | More scenes in the prompt, or a richer topic |
| Voice model won't download | Blocked network | It downloads once from Hugging Face; needs plain internet |
| Vietnamese narration sounds robotic | Fell back to an Edge vi-VN voice | Check the build log says `voice engine: vieneu`; if it fell back, fix the reason it did rather than retuning Edge |
| `External data path escapes model directory` | Hugging Face cached the ONNX weights as symlinks into its shared blob pool | The build repairs this itself and says so; it is safe to let it |
| Vietnamese scene fails with "no piper voice configured" | Edge and VieNeu both failed, and Piper has no Vietnamese model | Deliberate: an English Piper model reading Vietnamese returns confident gibberish. Fix the network, or add `voice.piper_voices.vi` |
| `quota exceeded` | Unlikely — uploads have their own ~100/day bucket | Wait for the daily reset |

---

## 10. Honest summary

There are two ways to run this, and both are built.

**Auto mode** (`run.py auto` on a schedule) publishes without you. The human
review step is replaced by an automated one: Claude checks each script for
invented facts and unsafe advice, and a similarity check compares every new
script against everything you have already published, so the channel cannot
drift into the templated sameness that YouTube demonetizes for. Anything
suspicious is held and emailed to you rather than published. This is a real
safety net, and it is weaker than your own eyes.

**Manual mode** (`run.py make`, then approve) costs about ten minutes per video
and catches things no automated check will — a mispronounced breed name, footage
that does not match the narration, a script that is technically accurate and
still boring.

The sensible path is to run auto mode but actually read the digest emails,
especially for the first month. If videos come back good week after week, keep
going. If the gate starts holding things, that is the system telling you
something upstream needs attention — and that signal is worth more than the ten
minutes it saves you.
