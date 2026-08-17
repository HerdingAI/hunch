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

`hunch setup` checks your hardware, tells you plainly what will work, indexes
your Documents, Downloads, Desktop and Pictures, and installs a background
timer. Nothing else is required — there is no daemon to configure and no cron
to write.

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

## Hardware

Document search runs on any CPU. Describing photos and transcribing audio are
much faster with an NVIDIA GPU of 4 GB or more; without one you can either
skip them or enable them with `hunch auth openrouter`, which explains what
gets uploaded before it stores anything.

Hunch is budgeted: the first index takes at most about five hours, and it
spends at most twenty cumulative minutes a day after that, tracked across the
hourly background timer rather than reset on every run, running only on
mains power and at idle priority.

## Privacy

No telemetry, ever. With the default local backend, no network requests are
made at all. Switching to OpenRouter sends the full text, photo, or audio
content of whatever it's enriching to OpenRouter's API — `hunch auth
openrouter` states this plainly and asks for confirmation before storing a
key. The index is a single SQLite file at `~/.local/share/hunch/index.db`,
created owner-only-readable — delete it and nothing remains.

## License

Apache-2.0. See `NOTICE` for the models and data this project uses.
