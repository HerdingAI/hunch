# Hunch

Find your files by what they *mean*, not what they are called.

Ask for "the lease agreement" and get the PDF whose text mentions a tenancy
term — plus the scan of the signed copy, and the voicemail where it was
discussed.

Everything runs on your machine by default. Nothing is uploaded.

## Install

```bash
sudo apt install pipx poppler-utils tesseract-ocr ffmpeg \
    python3-gi gir1.2-gtk-4.0 gir1.2-adw-1

pipx install --system-site-packages \
    "hunch-search[local,media] @ git+https://github.com/HerdingAI/hunch@main"

hunch setup
```

The first line installs what Hunch shells out to: `pdftotext` for PDFs,
`tesseract` for text inside images, `ffmpeg` for audio and video, and the
system GTK bindings the search window uses. `--system-site-packages` is what
lets the window find those bindings — without it the window won't open,
though everything else still works from the terminal.

The `local` extra brings the embedding and vision models (PyTorch, several
GB); `media` brings speech-to-text. Drop either if you don't want it — or
drop both and run `hunch auth openrouter` to do the heavy lifting in the
cloud instead.

`hunch setup` checks your hardware, tells you plainly what will work, picks
the folders to index — Documents, Downloads, Desktop and Pictures — and
starts a background timer that does the indexing from then on. Nothing else
is required: there is no daemon to configure and no cron to write.

Indexing runs in the background rather than making you wait, so results fill
in over the first few hours rather than appearing all at once. `hunch status`
shows how far along it is. To index right now instead of waiting for the
timer, run `hunch index`.

Press **Super+F** to search, or:

```bash
hunch search "that invoice from the plumber"
hunch status
hunch doctor
```

## What it understands

| | |
|---|---|
| Documents | PDF (including scanned, via OCR), Word, Excel, PowerPoint, ODF, ePub, email, plain text |
| Images | text in the image, capture date, place, camera, folder — plus an AI description of the contents |
| Audio & video | spoken words, transcribed |

By default only the first ten minutes of a recording is transcribed. Search
embeds roughly the first eight minutes' worth of words, so past that the audio
would be decoded at full cost and then discarded — on a 1,200-hour library that
is the difference between hours and days. A topic first raised late in a long
recording won't be findable by it; set `transcribe_max_seconds = 0` in the
config to transcribe recordings in full.

## Hardware

Document search runs on any CPU. Describing photos and transcribing audio are
much faster with an NVIDIA GPU of 4 GB or more; without one you can either
skip them or enable them with `hunch auth openrouter`, which explains what
gets uploaded before it stores anything.

Hunch is budgeted so it never takes the machine over: it spends at most about
five hours on the first pass, then at most twenty cumulative minutes a day,
tracked across the hourly background timer rather than reset on every run,
running only on mains power and at idle priority.

Those are spending limits, not completion times. Most home folders finish
well inside the first pass; a very large one may not, and then the remainder
is worked through at twenty minutes a day. Hunch tells you if that happens,
and `hunch status` shows which folders the backlog is in — the usual cause is
a folder full of machine-generated files that is better excluded than
indexed.

## Privacy

No telemetry, ever. With the default local backend, the contents of your
files never leave the machine — reading, transcribing and embedding all
happen locally.

The one thing that does use the network is downloading the models
themselves, from Hugging Face, the first time each is needed (they are not
shipped with the package — see `NOTICE`). That transfer sends no data about
you or your files, and once a model is cached nothing further is fetched.

Switching to OpenRouter is the deliberate exception: it sends the full text,
photo, or audio content of whatever it's enriching to OpenRouter's API.
`hunch auth openrouter` states this plainly and asks for confirmation before
storing a key.

The index is a single SQLite file at `~/.local/share/hunch/index.db`, created
owner-only-readable — delete it and nothing remains.

## License

Apache-2.0. See `NOTICE` for the models and data this project uses.
