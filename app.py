"""CT Dose AI — radiation dose for CT, predicted and optimised.

A Shiny for Python application written for clinical users: four screens, plain language,
everything technical available but out of the way.

Run it with either of:

    python app.py            # starts the server and opens the browser
    shiny run --reload app.py

Nothing leaves the computer: the model, the data and the logs all stay in this folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_widget

from ctdose import figures as fg
from ctdose.bandit import BanditConfig, DoseBandit, features as bandit_features
from ctdose.data import FEATURES, TARGETS, load_table
from ctdose.optimizer import OptimizerConfig, dose_curves, min_kv_for_bmi, recommend
from ctdose.phantom_bo import Campaign, dose_fn_from_predictor, suggest
from ctdose.predictor import DosePredictor, available_models

KV_CHOICES = [70, 80, 90, 100, 110, 120, 130, 140]
K_CARDIAC = 0.026                      # mSv per mGy·cm, cardiac-specific (Trattner et al. 2018)
CHEST_XRAY_MSV = 0.02
BACKGROUND_MSV_YEAR = 2.4

# ---------------------------------------------------------------------------- model loading
def _load_models() -> dict[str, DosePredictor]:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, path in available_models().items():
            try:
                out[name] = DosePredictor.load(path)
            except Exception as exc:                       # noqa: BLE001
                print(f"  ! could not load {path.name}: {exc}")
    return out


MODELS = _load_models()
if not MODELS:
    # Training at import time is right on a laptop and wrong on a server: hosts
    # give the process ~60 s to accept a connection, and a CV sweep takes longer
    # than that, so the deployment is killed mid-fit.  Only train when there is
    # genuinely nothing to load and we are not running under a host.
    _hosted = any(os.environ.get(v) for v in
                  ("RSTUDIO_CONNECT", "CONNECT_SERVER", "POSIT_PRODUCT", "SHINY_SERVER", "PORT"))
    if available_models() or _hosted or os.environ.get("CTDOSE_NO_AUTOTRAIN"):
        raise SystemExit(
            "No usable model could be loaded.\n"
            f"Model files present: {sorted(available_models()) or 'none'}\n"
            "If a file is listed above, it exists but failed to unpickle — see the error "
            "printed above this line. Otherwise commit models/*.joblib, or run `python train.py`."
        )
    print("No trained model found — running `python train.py` first.")
    from train import train_all
    train_all(repeats=3)
    MODELS = _load_models()

EXAMS = list(MODELS)


# ---------------------------------------------------------------------------- small helpers
def term(label: str, explanation: str) -> ui.Tag:
    """A term with a dotted underline; hovering or tapping explains it in plain words."""
    return ui.tooltip(
        ui.tags.span(label, class_="term"),
        explanation, placement="top")


def help_icon(explanation: str) -> ui.Tag:
    return ui.tooltip(ui.tags.span(icon_svg("circle-question"), class_="help-dot"),
                      explanation, placement="top")


def fmt(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}"


CSS = """
:root { --blue:#2a78d6; --dark:#1a4f94; --aqua:#1baf7a; --orange:#eb6834; --red:#e34948;
        --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8985; --line:#e3e3df; --surf:#f7f7f5; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); }
.navbar-brand { font-weight: 700; letter-spacing: -0.2px; }
.term { border-bottom: 1px dotted var(--blue); cursor: help; }
.help-dot svg { width: 13px; height: 13px; fill: var(--ink3); vertical-align: -1px; margin-left: 3px; cursor: help; }
.card { border-color: var(--line); }
.card-header { background: #fff; border-bottom: 1px solid var(--line); font-weight: 600; }
.big-number { font-size: 2.1rem; font-weight: 700; line-height: 1.05; }
.unit { font-size: 0.95rem; font-weight: 500; color: var(--ink2); }
.sub { font-size: 0.82rem; color: var(--ink2); }
.bslib-value-box .sub, .bslib-value-box .unit { color: rgba(255,255,255,0.88); }
.bslib-value-box .value-box-showcase svg { fill: rgba(255,255,255,0.92); }
.verdict { border-radius: 10px; padding: 12px 16px; font-size: 1.02rem; }
.verdict.ok   { background: #e8f6f0; border-left: 4px solid var(--aqua); }
.verdict.warn { background: #fdf0e8; border-left: 4px solid var(--orange); }
.verdict.bad  { background: #fdecec; border-left: 4px solid var(--red); }
.advice { background: #e8f6f0; border: 1px solid var(--aqua); border-radius: 12px; padding: 16px 18px; }
.advice h4 { margin: 0 0 6px; font-size: 1.12rem; }
.advice .set { font-size: 1.3rem; font-weight: 700; color: var(--dark); }
.step { background: var(--surf); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
.step b { color: var(--dark); }
.glossary td { padding: 7px 10px; vertical-align: top; border-bottom: 1px solid var(--line); font-size: 0.92rem; }
.glossary td:first-child { font-weight: 600; white-space: nowrap; color: var(--dark); }
.preset-btn { margin-right: 6px; margin-bottom: 6px; }
.small-note { font-size: 0.82rem; color: var(--ink2); }
.bslib-value-box .value-box-title { font-size: 0.82rem; }
"""


# ---------------------------------------------------------------------------- UI: screen 1
def patient_sidebar() -> ui.Tag:
    return ui.sidebar(
        ui.h5("This patient"),
        ui.input_slider("bmi", ui.span("BMI ", help_icon(
            "Weight divided by height squared. It cannot be changed — everything else adapts to it.")),
            min=15.0, max=55.0, value=27.0, step=0.1),
        ui.div(
            ui.input_action_button("p_slim", "Slim", class_="btn-sm btn-outline-secondary preset-btn"),
            ui.input_action_button("p_avg", "Average", class_="btn-sm btn-outline-secondary preset-btn"),
            ui.input_action_button("p_large", "Large", class_="btn-sm btn-outline-secondary preset-btn"),
            ui.input_action_button("p_xl", "Very large", class_="btn-sm btn-outline-secondary preset-btn"),
            class_="mb-3"),
        ui.hr(),
        ui.h5("Planned scan"),
        ui.input_select("kv", ui.span("Tube voltage (kV) ", help_icon(
            "How penetrating the beam is. Lower kV means much less dose and brighter iodine, "
            "but a noisier image in large patients.")),
            choices=[str(k) for k in KV_CHOICES], selected="120"),
        ui.input_slider("mas", ui.span("Tube current–time (mAs) ", help_icon(
            "How many X-rays are used. Dose is exactly proportional to it. The scanner usually "
            "picks this automatically from the patient's size.")),
            min=50, max=1600, value=750, step=5),
        ui.input_slider("length", ui.span("Scan length (cm) ", help_icon(
            "How much of the body is irradiated. Total dose is proportional to it.")),
            min=5.0, max=60.0, value=30.0, step=0.5),
        ui.hr(),
        ui.input_select("exam", "Examination", choices=EXAMS, selected=EXAMS[0]),
        ui.input_numeric("drl", ui.span("Reference level (mGy·cm) ", help_icon(
            "The benchmark your department compares against. 400 is the published value for "
            "coronary CT angiography — replace it with your local one.")), value=400, min=50, max=5000, step=10),
        ui.input_action_link("show_tour", "How do I use this?"),
        width=330, open="open",
    )


def screen_dose() -> ui.Tag:
    return ui.nav_panel(
        "1 · Dose",
        ui.layout_columns(
            ui.value_box("Dose per slice (CTDIvol)", ui.output_ui("vb_ctdi"), ui.output_ui("vb_ctdi_sub"),
                         showcase=icon_svg("layer-group"), theme="primary"),
            ui.value_box("Total dose (DLP)", ui.output_ui("vb_dlp"), ui.output_ui("vb_dlp_sub"),
                         showcase=icon_svg("chart-column"), theme="primary"),
            ui.value_box("Effective dose", ui.output_ui("vb_msv"), ui.output_ui("vb_msv_sub"),
                         showcase=icon_svg("person"), theme="secondary"),
            col_widths=[4, 4, 4], fill=False),
        ui.card(
            ui.card_header("Where this scan sits"),
            output_widget("fig_bar"),
            ui.output_ui("verdict_box"),
            ui.output_ui("domain_note"),
        ),
        ui.layout_columns(
            ui.card(ui.card_header(ui.span("How the dose follows the settings ",
                                           help_icon("Move the sliders on the left — the orange dot moves with "
                                                     "them. Each line is one tube voltage."))),
                    output_widget("fig_dose_mas")),
            ui.card(ui.card_header("What these numbers mean for the patient"),
                    ui.output_ui("patient_context")),
            col_widths=[7, 5]),
        ui.card(
            ui.card_header("Where the prediction comes from"),
            ui.output_ui("provenance"),
        ),
    )


def screen_lower() -> ui.Tag:
    return ui.nav_panel(
        "2 · Lower it",
        ui.output_ui("advice_box"),
        ui.layout_columns(
            ui.card(ui.card_header(ui.span("The trade-off ", help_icon(
                "Left and up is better: less dose, same quality. The star is the suggestion."))),
                output_widget("fig_tradeoff")),
            ui.card(ui.card_header(ui.span("Will the image still be diagnostic? ", help_icon(
                "Starts as physics. Each exam a radiologist scores in the Advanced screen moves this curve "
                "towards your own practice."))),
                output_widget("fig_accept")),
            col_widths=[6, 6]),
        ui.accordion(
            ui.accordion_panel(
                "Try a setting yourself",
                ui.layout_columns(
                    ui.input_select("w_kv", "Tube voltage (kV)", choices=[str(k) for k in KV_CHOICES],
                                    selected="100"),
                    ui.input_slider("w_mas", "mAs", min=50, max=1600, value=750, step=5),
                    ui.input_slider("w_len", "Scan length (cm)", min=5.0, max=60.0, value=30.0, step=0.5),
                    col_widths=[3, 5, 4]),
                ui.output_ui("whatif_out"), value="whatif"),
            ui.accordion_panel(
                "Every option, and the limits used",
                ui.layout_columns(
                    ui.input_selectize("kv_opts", "kV your scanner offers",
                                       choices=[str(k) for k in KV_CHOICES],
                                       selected=["80", "100", "120", "140"], multiple=True),
                    ui.input_numeric("mas_max", "Largest mAs the tube allows", value=1500, min=100, max=5000,
                                     step=50),
                    ui.input_slider("keep_q", "Image quality to keep (%)", min=70, max=110, value=100, step=5),
                    ui.input_switch("bmi_rule", "Minimum-kV rule for BMI", value=True),
                    col_widths=[3, 3, 4, 2]),
                output_widget("fig_options"),
                ui.output_data_frame("tbl_options"),
                ui.output_ui("reasons"), value="options"),
            open=False, multiple=False),
    )


def screen_before() -> ui.Tag:
    return ui.nav_panel(
        "3 · Before the scan",
        ui.card(
            ui.card_header("Only the BMI is known yet"),
            ui.markdown(
                "Screen 1 needs the mAs, which the scanner picks once the patient is on the table. "
                "Here the app predicts that choice from the BMI, so you can estimate the dose in advance — "
                "for consent, planning or an audit. **The ranges are wider, and they should be.**"),
            ui.layout_columns(
                ui.input_select("pre_kv", "Tube voltage you plan to use",
                                choices=[str(k) for k in KV_CHOICES], selected="120"),
                ui.input_switch("pre_len_known", "I already know the scan length", value=False),
                ui.input_slider("pre_len", "Scan length (cm)", min=5.0, max=60.0, value=30.0, step=0.5),
                col_widths=[3, 3, 6]),
        ),
        ui.layout_columns(
            ui.value_box("mAs the scanner will likely choose", ui.output_ui("pre_mas"),
                         ui.output_ui("pre_mas_sub"), showcase=icon_svg("bolt"), theme="primary"),
            ui.value_box("Dose per slice", ui.output_ui("pre_ctdi"), ui.output_ui("pre_ctdi_sub"),
                         showcase=icon_svg("layer-group"), theme="primary"),
            ui.value_box("Total dose", ui.output_ui("pre_dlp"), ui.output_ui("pre_dlp_sub"),
                         showcase=icon_svg("chart-column"), theme="primary"),
            col_widths=[4, 4, 4], fill=False),
        ui.layout_columns(
            ui.card(ui.card_header("What the scanner chose for past patients"), output_widget("fig_pre_mas")),
            ui.card(ui.card_header("Scan length against BMI"), output_widget("fig_pre_len")),
            col_widths=[6, 6]),
    )


def screen_learn() -> ui.Tag:
    return ui.nav_panel(
        "Learn",
        ui.layout_columns(
            ui.card(
                ui.card_header("The three things this app does"),
                ui.div(ui.markdown(
                    "**1 · It predicts the dose.** Your scanner's own dose law was fitted on past "
                    "examinations, so given kV, mAs and scan length it knows what the console will report, "
                    "to well under one percent."), class_="step"),
                ui.div(ui.markdown(
                    "**2 · It suggests a lower-dose setting.** It searches tube voltage, tube current and "
                    "scan length for the combination that delivers *the same image quality* for less dose. "
                    "It never changes the BMI, and it never proposes a voltage that is too low for the "
                    "patient's size."), class_="step"),
                ui.div(ui.markdown(
                    "**3 · It learns.** Measure image quality on a phantom, or score real images 1–4, and "
                    "the suggestions move from textbook physics towards your own department's practice."),
                    class_="step"),
            ),
            ui.card(
                ui.card_header("Why lower kV saves so much dose"),
                ui.markdown(
                    "Two facts do all the work:\n\n"
                    "- Image **noise** falls as the square root of dose. Four times the dose halves the noise.\n"
                    "- Iodine **contrast rises as the voltage falls**, because the beam energy moves closer to "
                    "the energy iodine absorbs best.\n\n"
                    "So dropping from 120 to 100 kV makes the vessel brighter, which leaves room to cut the "
                    "dose by about 40% and still see exactly as well. That is the whole idea behind screen 2 — "
                    "and the reason the suggestion sits *left* of your scan on the trade-off chart, at the "
                    "same height."),
                ui.input_action_button("explain_cnr", "What is contrast-to-noise?",
                                       class_="btn-sm btn-outline-primary"),
            ),
            col_widths=[6, 6]),
        ui.card(
            ui.card_header("The words on screen"),
            ui.HTML("""
<table class="glossary" style="width:100%">
<tr><td>CTDIvol</td><td>Dose per slice, in mGy. How strong the radiation is at each slice. It does not depend on how long the scan is.</td></tr>
<tr><td>DLP</td><td>Total dose, in mGy·cm. CTDIvol multiplied by the irradiated length. This is what reference levels are written in.</td></tr>
<tr><td>Effective dose (mSv)</td><td>A risk-weighted whole-body estimate. For cardiac CT, roughly DLP × 0.026.</td></tr>
<tr><td>Reference level</td><td>A benchmark, not a limit. If your median sits above it, the protocol deserves review.</td></tr>
<tr><td>kV</td><td>Tube voltage: how penetrating the beam is. The strongest lever on dose in contrast-enhanced CT.</td></tr>
<tr><td>mAs</td><td>Tube current × rotation time: how many X-rays are used. Dose is exactly proportional to it.</td></tr>
<tr><td>CNR</td><td>Contrast-to-noise ratio: how far the iodine-filled vessel stands out from the grain. What must not get worse.</td></tr>
<tr><td>Phantom</td><td>A plastic test object scanned instead of a patient, so image quality can be measured without irradiating anyone.</td></tr>
<tr><td>Prediction range</td><td>The band around each number. The true value falls inside it about nine times out of ten.</td></tr>
</table>"""),
        ),
        ui.card(
            ui.card_header("Honest limits"),
            ui.markdown(
                "- This is **decision support for research use**, not a medical device, and it has not been "
                "clinically validated. Any protocol change needs the medical physicist and the responsible "
                "radiologist.\n"
                "- The fitted law belongs to **one scanner**. Another scanner needs its own data.\n"
                "- Voltages that never appear in your data (80 and 100 kV here) are an **extrapolation** from "
                "physics. The app says so whenever it does this — confirm on a phantom before clinical use.\n"
                "- The image-quality model is first-order. It ignores beam hardening, iterative reconstruction, "
                "body shape and cardiac motion, which is exactly why the two learning loops exist."),
        ),
    )


def screen_advanced() -> ui.Tag:
    return ui.nav_panel(
        "Advanced",
        ui.markdown("For the medical physicist and whoever maintains the model. Nothing here is needed for "
                    "everyday use."),
        ui.accordion(
            ui.accordion_panel("Model card and accuracy", ui.output_ui("model_card"),
                               ui.output_data_frame("tbl_cv"),
                               ui.layout_columns(output_widget("fig_oof_ctdi"), output_widget("fig_oof_dlp"),
                                                 col_widths=[6, 6]), value="card"),
            ui.accordion_panel("Data quality screen", ui.output_ui("quality_summary"),
                               ui.output_data_frame("tbl_quality"), value="quality"),
            ui.accordion_panel(
                "Phantom testing (Bayesian optimisation)",
                ui.markdown("Scan the phantom at the setting below, measure the CNR in the iodine insert, "
                            "and type it in. The app proposes the next setting and stops when nothing better "
                            "is expected — usually after 8 to 20 scans."),
                ui.layout_columns(
                    ui.input_select("ph_campaign", "Campaign", choices=["(new)"], selected="(new)"),
                    ui.input_action_button("ph_new", "Start a campaign", class_="btn-outline-primary"),
                    ui.input_action_button("ph_demo", "Run a simulated demo", class_="btn-outline-secondary"),
                    col_widths=[5, 3, 4]),
                ui.output_ui("ph_status"),
                ui.layout_columns(
                    ui.input_select("ph_kv", "kV scanned", choices=[str(k) for k in KV_CHOICES], selected="120"),
                    ui.input_numeric("ph_mas", "mAs scanned", value=750, min=10, max=3000, step=5),
                    ui.input_numeric("ph_cnr", "CNR you measured", value=0.0, min=0.0, max=1000.0, step=0.1),
                    ui.input_action_button("ph_add", "Add measurement", class_="btn-primary"),
                    col_widths=[3, 3, 3, 3]),
                output_widget("fig_phantom"), value="phantom"),
            ui.accordion_panel(
                "Learning from radiologists",
                ui.markdown("One proposal per patient, approved by a radiologist, then scored 1–4 once the "
                            "images are read. Scores teach the acceptability curve on screen 2."),
                ui.layout_columns(
                    ui.input_action_button("bd_propose", "Propose a protocol for this patient",
                                           class_="btn-primary"),
                    ui.input_select("bd_score", "Diagnostic quality of the last scan",
                                    choices={"": "—", "4": "4 · excellent", "3": "3 · diagnostic",
                                             "2": "2 · limited", "1": "1 · non-diagnostic"}, selected=""),
                    ui.input_action_button("bd_record", "Record it", class_="btn-outline-primary"),
                    col_widths=[4, 4, 4]),
                ui.output_ui("bd_out"),
                ui.output_data_frame("tbl_bandit"), value="bandit"),
            ui.accordion_panel(
                "Many scans at once",
                ui.input_file("batch_file", "CSV, Excel or PDF with BMI, kV, mAs and scan length",
                              accept=[".csv", ".xlsx", ".xls", ".pdf"]),
                ui.output_ui("batch_summary"),
                ui.output_data_frame("tbl_batch"),
                ui.download_button("batch_dl", "Download the predictions", class_="btn-outline-primary"),
                value="batch"),
            open=False, multiple=False),
    )


app_ui = ui.page_navbar(
    screen_dose(), screen_lower(), screen_before(), screen_learn(), screen_advanced(),
    title=ui.span(icon_svg("radiation"), "  CT Dose AI"),
    sidebar=patient_sidebar(),
    header=ui.tags.style(CSS),
    footer=ui.div(ui.markdown(
        "Decision support for research use — not a medical device. Every protocol change needs the medical "
        "physicist and the responsible radiologist."), class_="small-note p-3"),
    fillable=False, id="nav",
)


# ============================================================================ server
def server(input, output, session):  # noqa: C901 - a UI server is naturally wide
    phantom_version = reactive.value(0)
    bandit_version = reactive.value(0)
    last_proposal = reactive.value(None)

    # ------------------------------------------------------------------ core reactives
    @reactive.calc
    def pred() -> DosePredictor:
        return MODELS[input.exam()]

    @reactive.calc
    def kv() -> float:
        return float(input.kv())

    @reactive.calc
    def current() -> dict:
        return pred().predict_one(float(input.bmi()), kv(), float(input.mas()), float(input.length()))

    @reactive.calc
    def gamma() -> float:
        """Iodine exponent: the phantom-calibrated value when a campaign exists, else the literature one."""
        phantom_version()
        f = dose_fn_from_predictor(pred())
        best = None
        for name, path in Campaign.list().items():
            try:
                c = Campaign.load(path)
                if len(c.scans) >= 4 and c.target():
                    r = suggest(c, f)
                    if r.surrogate is not None:
                        best = float(r.surrogate.gamma)
            except Exception:                              # noqa: BLE001
                continue
        return best if best else 1.35

    @reactive.calc
    def cfg() -> OptimizerConfig:
        kvs = tuple(sorted({int(k) for k in input.kv_opts()} | {int(kv())})) or (120,)
        return OptimizerConfig(kv_options=kvs, cnr_target=float(input.keep_q()) / 100, iodine_gamma=gamma(),
                               mas_max=float(input.mas_max()), length_cuts_cm=(0, 2),
                               apply_bmi_kv_rule=bool(input.bmi_rule()))

    @reactive.calc
    def rec() -> dict:
        return recommend(pred(), float(input.bmi()), kv(), float(input.mas()), float(input.length()), cfg())

    @reactive.calc
    def bandit() -> DoseBandit:
        bandit_version()
        bcfg = BanditConfig(gamma=gamma(), kv_options=tuple(sorted({int(k) for k in input.kv_opts()})) or (120,),
                            apply_bmi_kv_rule=bool(input.bmi_rule()))
        return DoseBandit(pred(), bcfg, log=DoseBandit.load_log(input.exam()), seed=0)

    # ------------------------------------------------------------------ presets & sync
    @reactive.effect
    @reactive.event(input.p_slim)
    def _():
        ui.update_slider("bmi", value=22.0)

    @reactive.effect
    @reactive.event(input.p_avg)
    def _():
        ui.update_slider("bmi", value=27.0)

    @reactive.effect
    @reactive.event(input.p_large)
    def _():
        ui.update_slider("bmi", value=34.0)

    @reactive.effect
    @reactive.event(input.p_xl)
    def _():
        ui.update_slider("bmi", value=42.0)

    @reactive.effect
    def _sync_defaults():
        """Start from the median patient of this examination, and keep the what-if panel in step."""
        d = pred().train_df.median(numeric_only=True)
        ui.update_slider("mas", value=float(round(d.mAs)))
        ui.update_slider("length", value=float(round(d.scan_length_cm, 1)))
        ui.update_slider("bmi", value=float(round(d.BMI, 1)))

    @reactive.effect
    def _sync_whatif():
        ui.update_slider("w_mas", value=float(input.mas()))
        ui.update_slider("w_len", value=float(input.length()))

    @reactive.effect
    @reactive.event(input.show_tour)
    def _tour():
        ui.modal_show(ui.modal(
            ui.markdown(
                "**1 · Describe the patient and the planned scan** in the panel on the left. "
                "The four values are the ones your console shows. Not sure? Use the Slim / Average / Large "
                "buttons.\n\n"
                "**2 · Screen 1 tells you the dose** and whether it sits above or below your reference level, "
                "with the equivalent in millisieverts and in chest X-rays.\n\n"
                "**3 · Screen 2 gives you one change to make** — a tube voltage and an mAs that deliver the "
                "same image quality for less dose — and shows you the trade-off behind it.\n\n"
                "**4 · Screen 3 does all of this before the scan**, when only the BMI is known.\n\n"
                "Any underlined word has an explanation if you hover over it. The *Learn* screen explains "
                "the ideas, and *Advanced* holds the physicist's material."),
            title="How to use CT Dose AI", easy_close=True, size="l"))

    @reactive.effect
    @reactive.event(input.explain_cnr)
    def _cnr():
        ui.modal_show(ui.modal(
            ui.markdown(
                "**Contrast-to-noise ratio** answers one question: can you actually see the thing you are "
                "looking for?\n\n"
                "- **Contrast** — how different the vessel is from its surroundings. An iodine-filled coronary "
                "artery sits at about +350 HU against muscle at about +50, so the contrast is roughly 300 HU.\n"
                "- **Noise** — the random grain. Measured as the standard deviation inside a uniform region, "
                "perhaps 25 HU.\n"
                "- **CNR = 300 / 25 = 12.** The signal is twelve times larger than the fluctuation, so the "
                "vessel is unmistakable.\n\n"
                "The app never shows an absolute CNR, because it depends on your contrast injection and your "
                "reconstruction. It shows CNR **relative to the scan you entered**: 100% means no loss at all, "
                "85% means the vessel stands out about 15% less clearly.\n\n"
                "Why it is the right thing to hold constant: lowering kV raises noise *and* raises iodine "
                "contrast. Only the ratio tracks what the radiologist actually experiences."),
            title="Contrast-to-noise, in plain terms", easy_close=True, size="l"))

    # ------------------------------------------------------------------ screen 1 outputs
    @render.ui
    def vb_ctdi():
        r = current()
        return ui.HTML(f"{r['CTDIvol']:.1f} <span class='unit'>mGy</span>")

    @render.ui
    def vb_ctdi_sub():
        r = current()
        return ui.span(f"range {r['CTDIvol_lo']:.1f} – {r['CTDIvol_hi']:.1f}", class_="sub")

    @render.ui
    def vb_dlp():
        r = current()
        return ui.HTML(f"{fmt(r['DLP'])} <span class='unit'>mGy·cm</span>")

    @render.ui
    def vb_dlp_sub():
        r = current()
        pct = (r["DLP"] / float(input.drl()) - 1) * 100
        return ui.span(f"range {fmt(r['DLP_lo'])} – {fmt(r['DLP_hi'])} · {pct:+.0f}% vs reference", class_="sub")

    @render.ui
    def vb_msv():
        return ui.HTML(f"≈ {current()['DLP'] * K_CARDIAC:.1f} <span class='unit'>mSv</span>")

    @render.ui
    def vb_msv_sub():
        return ui.span("cardiac factor 0.026 mSv per mGy·cm", class_="sub")

    @render_widget
    def fig_bar():
        r = current()
        return fg.dose_bar(r["DLP"], r["DLP_lo"], r["DLP_hi"], float(input.drl()))

    @render.ui
    def verdict_box():
        r, drl = current(), float(input.drl())
        ratio = r["DLP"] / drl
        if ratio <= 1.0:
            cls, msg = "ok", (f"This scan is **below** your reference level of {drl:g} mGy·cm. "
                              "Screen 2 can still look for savings.")
        elif ratio <= 1.5:
            cls, msg = "warn", (f"This scan is **{ratio:.1f}× your reference level** of {drl:g} mGy·cm. "
                                "Worth reviewing — see screen 2.")
        else:
            cls, msg = "bad", (f"This scan is **{ratio:.1f}× your reference level** of {drl:g} mGy·cm. "
                               "Screen 2 shows what to change.")
        return ui.div(ui.markdown(msg), class_=f"verdict {cls} mt-2")

    @render.ui
    def domain_note():
        r = current()
        if r["in_domain"]:
            return None
        return ui.div(ui.markdown(f"⚠ {r['domain_note']}. The physics still holds, but no past examination in "
                                  "your data looks like this, so treat it as an extrapolation."),
                      class_="verdict warn mt-2")

    @render_widget
    def fig_dose_mas():
        r, p = current(), pred()
        kvs = sorted({80, 100, 120, 140, int(kv())})
        curves = dose_curves(p, float(input.bmi()), float(input.length()), kvs=kvs,
                             mas_range=(50, max(1500, float(input.mas()) * 1.2)))
        return fg.dose_vs_mas(curves, kvs, float(input.mas()), r["DLP"], r["DLP_lo"], r["DLP_hi"],
                              "DLP", float(input.drl()))

    @render.ui
    def patient_context():
        r = current()
        msv = r["DLP"] * K_CARDIAC
        return ui.markdown(
            f"An effective dose of about **{msv:.1f} mSv** is roughly:\n\n"
            f"- **{msv / CHEST_XRAY_MSV:,.0f} chest X-rays**\n"
            f"- **{msv / BACKGROUND_MSV_YEAR:.1f} years** of natural background radiation\n\n"
            "These are rules of thumb for conversation with a patient, not a risk calculation. Effective dose "
            "is a population-average construct: it is not the dose to this particular person, and it says "
            "nothing on its own about whether the examination is justified — a diagnostic scan that answers "
            "the question is almost always worth it.")

    @render.ui
    def provenance():
        p = pred()
        q = getattr(p, "quality", {}) or {}
        k = float(np.exp(p.physics_ctdi.intercept_))
        n = p.physics_ctdi.kv_exponent_
        ctdi_err = p.cv_summary["CTDIvol"].loc[p.cv_summary["CTDIvol"].selected, "MAPE_%"].iloc[0]
        dlp_err = p.cv_summary["DLP"].loc[p.cv_summary["DLP"].selected, "MAPE_%"].iloc[0]
        extra = ""
        if q.get("n_error"):
            extra = (f" out of {q['n_rows']} in the file — {q['n_error']} were held back because their numbers "
                     f"are not physically consistent (see *Advanced → Data quality screen*)")
        return ui.markdown(
            f"Your scanner's own dose law was fitted on **{p.n_train} examinations**{extra}:\n\n"
            f"`dose per slice = {k:.4f} × mAs × (kV / 120) ^ {n:.2f}`  ·  `total dose = dose per slice × length`\n\n"
            f"Typical error in cross-validation: **{ctdi_err:.1f}%** for the dose per slice and "
            f"**{dlp_err:.1f}%** for the total. It was chosen over eight alternatives, including a neural "
            "network — with a known physical law and a few dozen examinations, the physics wins.")

    # ------------------------------------------------------------------ screen 2 outputs
    @render.ui
    def advice_box():
        r = rec()
        best, cur = r["best"], r["current"]
        if not best or best["DLP"] >= cur["DLP"] * 0.99:
            return ui.div(ui.markdown(
                "With the limits you have set, nothing lowers the dose while keeping image quality. "
                "Open *Every option, and the limits used* below and allow a lower kV, or shorten the scan."),
                class_="verdict warn")
        extra = (f", and shorten the scan to <b>{best['scan_length_cm']:.1f} cm</b>"
                 if abs(best["scan_length_cm"] - float(input.length())) > 0.05 else "")
        note = ""
        if best["kVp"] not in (pred().audit.get("kVp_levels") or []):
            note = (f"<div class='small-note mt-2'>⚠ {best['kVp']:g} kV does not appear in your data. This "
                    f"comes from the fitted kV law — confirm it on a phantom before clinical use.</div>")
        return ui.div(ui.HTML(f"""
<div class="advice">
  <h4>Suggested change</h4>
  <div class="set">{best['kVp']:.0f} kV &nbsp;·&nbsp; {best['mAs']:.0f} mAs{extra}</div>
  <div class="sub mb-2">from {kv():.0f} kV, {float(input.mas()):.0f} mAs, {float(input.length()):.1f} cm</div>
  <div class="row mt-2">
    <div class="col-sm-4"><div class="sub">Dose per slice</div>
      <div class="big-number" style="color:var(--dark)">{best['CTDIvol']:.1f}
      <span class="unit">mGy &nbsp;({best['CTDI_change_pct']:+.0f}%)</span></div></div>
    <div class="col-sm-4"><div class="sub">Total dose</div>
      <div class="big-number" style="color:var(--dark)">{fmt(best['DLP'])}
      <span class="unit">mGy·cm &nbsp;({best['DLP_change_pct']:+.0f}%)</span></div></div>
    <div class="col-sm-4"><div class="sub">Image quality kept</div>
      <div class="big-number" style="color:var(--aqua)">{best['CNR_rel']*100:.0f}
      <span class="unit">% of now</span></div></div>
  </div>
  {note}
</div>"""), class_="mb-3")

    @render_widget
    def fig_tradeoff():
        r = rec()
        cur = r["current"]
        kvs = sorted({int(k) for k in input.kv_opts()} | {int(kv())})
        return fg.quality_vs_dose(kv(), cur["CTDIvol"], cur["DLP"] / cur["CTDIvol"], kvs, gamma(),
                                  float(input.keep_q()), r["best"])

    @render_widget
    def fig_accept():
        r, b = rec(), bandit()
        cur = r["current"]
        kvs = sorted({int(k) for k in input.kv_opts()} | {int(kv())})
        return fg.acceptability(b.model, bandit_features, float(input.bmi()), b.ctdi_ref(float(input.bmi())),
                                cur["CTDIvol"], cur["DLP"] / cur["CTDIvol"], kvs, b.cfg.target_p, r["best"],
                                int(len(b.scored())))

    @render.ui
    def whatif_out():
        p = pred()
        base = current()
        alt = p.predict_one(float(input.bmi()), float(input.w_kv()), float(input.w_mas()), float(input.length()))
        dlp = alt["DLP"] * float(input.w_len()) / float(input.length())
        q = (kv() / float(input.w_kv())) ** gamma() * np.sqrt(alt["CTDIvol"] / base["CTDIvol"]) * 100
        warn = ""
        if float(input.w_kv()) < min_kv_for_bmi(float(input.bmi()), cfg()):
            warn = (f"<div class='verdict warn mt-2'>{float(input.w_kv()):.0f} kV is below the minimum "
                    f"suggested for BMI {float(input.bmi()):.1f} — expect a noisy image.</div>")
        colour = "var(--aqua)" if q >= 98 else ("var(--orange)" if q >= 85 else "var(--red)")
        return ui.HTML(f"""
<div class="row mt-2">
  <div class="col-sm-4"><div class="sub">Dose per slice</div><div class="big-number">{alt['CTDIvol']:.1f}
    <span class="unit">mGy ({(alt['CTDIvol']/base['CTDIvol']-1)*100:+.0f}%)</span></div></div>
  <div class="col-sm-4"><div class="sub">Total dose</div><div class="big-number">{fmt(dlp)}
    <span class="unit">mGy·cm ({(dlp/base['DLP']-1)*100:+.0f}%)</span></div></div>
  <div class="col-sm-4"><div class="sub">Image quality</div>
    <div class="big-number" style="color:{colour}">{q:.0f}<span class="unit">% of now</span></div></div>
</div>{warn}""")

    @render_widget
    def fig_options():
        r = rec()
        cand = r["candidates"]
        cand = cand[(cand.DLP <= r["current"]["DLP"] * 1.01) | (cand.kind == "current")].sort_values("DLP")
        return fg.options_bar(cand, r["current"]["DLP"], float(input.drl()))

    @render.data_frame
    def tbl_options():
        r = rec()
        c = r["candidates"]
        c = c[(c.DLP <= r["current"]["DLP"] * 1.01) | (c.kind == "current")].sort_values("DLP")
        out = pd.DataFrame({
            "Option": c.option, "kV": c.kVp.round(0), "mAs": c.mAs.round(0),
            "Length (cm)": c.scan_length_cm.round(1), "Per slice (mGy)": c.CTDIvol.round(1),
            "Total (mGy·cm)": c.DLP.round(0), "Change": c.DLP_change_pct.round(0).astype(int).astype(str) + "%",
            "Quality": (c.CNR_rel * 100).round(0).astype(int).astype(str) + "%",
            "Allowed": np.where(c.feasible, "yes", "no"), "Notes": c.flags})
        return render.DataGrid(out, height="280px", width="100%")

    @render.ui
    def reasons():
        return ui.div(ui.markdown("\n".join("- " + m for m in rec()["messages"])), class_="small-note mt-2")

    # ------------------------------------------------------------------ screen 3 outputs
    @reactive.calc
    def prescan() -> dict | None:
        p = pred()
        if getattr(p, "prescan", None) is None:
            return None
        return p.predict_prescan(float(input.bmi()), kVp=float(input.pre_kv()),
                                 scan_length_cm=float(input.pre_len()) if input.pre_len_known() else None)

    @render.ui
    def pre_mas():
        r = prescan()
        return ui.HTML("—" if not r else f"{r['mAs']:.0f} <span class='unit'>mAs</span>")

    @render.ui
    def pre_mas_sub():
        r = prescan()
        return ui.span("" if not r else f"range {r['mAs_lo']:.0f} – {r['mAs_hi']:.0f}", class_="sub")

    @render.ui
    def pre_ctdi():
        r = prescan()
        return ui.HTML("—" if not r else f"{r['CTDIvol']:.1f} <span class='unit'>mGy</span>")

    @render.ui
    def pre_ctdi_sub():
        r = prescan()
        return ui.span("" if not r else f"range {r['CTDIvol_lo']:.1f} – {r['CTDIvol_hi']:.1f}", class_="sub")

    @render.ui
    def pre_dlp():
        r = prescan()
        return ui.HTML("—" if not r else f"{fmt(r['DLP'])} <span class='unit'>mGy·cm</span>")

    @render.ui
    def pre_dlp_sub():
        r = prescan()
        if not r:
            return ui.span("this model has no pre-scan stage — run train.py", class_="sub")
        return ui.span(f"range {fmt(r['DLP_lo'])} – {fmt(r['DLP_hi'])}", class_="sub")

    @render_widget
    def fig_pre_mas():
        p, r = pred(), prescan()
        req(r)
        return fg.prescan_curve(p.prescan.curve(), p.train_df, float(input.bmi()), r["mAs"], "mAs",
                                "mAs the scanner chose, against BMI", "mAs")

    @render_widget
    def fig_pre_len():
        p, r = pred(), prescan()
        req(r)
        return fg.prescan_curve(p.prescan.curve(), p.train_df, float(input.bmi()), r["scan_length_cm"],
                                "scan_length_cm", "Scan length, against BMI", "cm")

    # ------------------------------------------------------------------ advanced: model card
    @render.ui
    def model_card():
        p = pred()
        return ui.markdown(
            f"**{input.exam()}** · trained {p.trained_at} on {p.n_train} examinations "
            f"({p.audit['n_unique']} unique) · scikit-learn {getattr(p, 'sklearn_version', '?')}\n\n"
            f"Selected: " + ", ".join(f"**{t}** → {n}" for t, n in p.selected.items()) +
            f"\n\n90% prediction intervals: CTDIvol × [{np.exp(p.calibrators['CTDIvol'].q_lo_):.3f}, "
            f"{np.exp(p.calibrators['CTDIvol'].q_hi_):.3f}], DLP × "
            f"[{np.exp(p.calibrators['DLP'].q_lo_):.3f}, {np.exp(p.calibrators['DLP'].q_hi_):.3f}]")

    @render.data_frame
    def tbl_cv():
        p = pred()
        frames = []
        for t in TARGETS:
            s = p.cv_summary[t][["model", "MAPE_%", "MedAPE_%", "R2", "MaxAPE_%", "selected"]].copy()
            s.insert(0, "target", t)
            frames.append(s)
        out = pd.concat(frames)
        out["MAPE_%"] = out["MAPE_%"].round(2)
        out["MedAPE_%"] = out["MedAPE_%"].round(2) if "MedAPE_%" in out else np.nan
        out["R2"] = out["R2"].round(4)
        out["MaxAPE_%"] = out["MaxAPE_%"].round(1)
        out["selected"] = np.where(out["selected"], "◀ selected", "")
        return render.DataGrid(out, height="330px", width="100%")

    @render_widget
    def fig_oof_ctdi():
        p = pred()
        return fg.predicted_vs_measured(p.train_df.CTDIvol.to_numpy(),
                                        p.oof["CTDIvol"][p.selected["CTDIvol"]].mean(axis=0), "CTDIvol")

    @render_widget
    def fig_oof_dlp():
        p = pred()
        return fg.predicted_vs_measured(p.train_df.DLP.to_numpy(),
                                        p.oof["DLP"][p.selected["DLP"]].mean(axis=0), "DLP")

    # ------------------------------------------------------------------ advanced: quality
    @render.ui
    def quality_summary():
        q = getattr(pred(), "quality", {}) or {}
        if not q:
            return ui.markdown("This model predates the quality screen. Run `python train.py` to add it.")
        lp = q.get("length_predictable", {})
        verdict = lp.get("verdict", "")
        sc = q.get("systematic_cluster")
        extra = ""
        if sc:
            extra = (f"\n\n**{sc['n']} examinations share a {sc['mean_yield_dev_pct']:+.0f}% dose per mAs** "
                     f"(rows {sc['rows']}). {sc['note']}")
        return ui.markdown(
            f"{q['n_rows']} examinations in the file · **{q.get('training_rows', '?')} used for training** · "
            f"{q['n_review']} flagged for review · {q['n_error']} held out as probable errors.\n\n"
            "The screen is physics-based: CTDIvol/mAs must match the scanner constant for that voltage, and "
            "DLP/CTDIvol must be a plausible irradiated length. Nothing is deleted from your file."
            + extra + (f"\n\nCan the irradiated length be recovered from the four inputs? {verdict}" if verdict else ""))

    @render.data_frame
    def tbl_quality():
        q = getattr(pred(), "quality", {}) or {}
        rows = q.get("flagged", [])
        if not rows:
            return render.DataGrid(pd.DataFrame({"result": ["every row is physically consistent"]}))
        f = pd.DataFrame(rows)
        out = pd.DataFrame({
            "Row": f.row, "Flag": f.flag, "BMI": f.BMI, "kV": f.kVp.round(0), "mAs": f.mAs.round(0),
            "CTDIvol": f.CTDIvol.round(1), "DLP": f.DLP.round(0),
            "Dose/mAs vs constant": f.yield_dev_pct.round(1).astype(str) + "%",
            "Implied length (cm)": f.implied_length_cm.round(1), "Why": f.reason})
        return render.DataGrid(out, height="260px", width="100%")

    # ------------------------------------------------------------------ advanced: phantom
    @reactive.effect
    def _refresh_campaigns():
        phantom_version()
        names = list(Campaign.list())
        ui.update_select("ph_campaign", choices=["(new)"] + names,
                         selected=names[-1] if names else "(new)")

    @reactive.calc
    def campaign() -> Campaign | None:
        phantom_version()
        name = input.ph_campaign()
        paths = Campaign.list()
        if name in paths:
            return Campaign.load(paths[name])
        return None

    @reactive.effect
    @reactive.event(input.ph_new)
    def _new_campaign():
        d = pred().train_df.median(numeric_only=True)
        c = Campaign(name=f"{input.exam()}_{len(Campaign.list())+1}", exam=input.exam(),
                     kv_options=sorted({int(k) for k in input.kv_opts()}) or [80, 100, 120],
                     ref_kVp=float(kv()), ref_mAs=float(round(d.mAs / 5) * 5))
        c.save()
        phantom_version.set(phantom_version() + 1)
        ui.notification_show("Campaign created. Scan the phantom at the setting shown, then type the CNR.",
                             duration=6)

    @reactive.effect
    @reactive.event(input.ph_demo)
    def _demo_campaign():
        from ctdose.phantom_bo import run_simulation
        d = pred().train_df.median(numeric_only=True)
        sim = run_simulation(dose_fn_from_predictor(pred()), seed=int(np.random.randint(1000)),
                             ref=(120, float(round(d.mAs / 5) * 5)))
        c = sim["campaign"]
        c.name, c.exam, c.phantom = "DEMO_simulated", input.exam(), "synthetic phantom (demonstration)"
        c.save()
        phantom_version.set(phantom_version() + 1)
        ui.notification_show(
            f"Demo finished in {sim['n_scans']} scans: {sim['recommended']['kVp']:.0f} kV / "
            f"{sim['recommended']['mAs']:.0f} mAs ({sim['recommended']['CTDIvol']:.1f} mGy) against a starting "
            f"{sim['ref_dose']:.1f} mGy.", duration=10)

    @reactive.effect
    @reactive.event(input.ph_add)
    def _add_measurement():
        c = campaign()
        if c is None or float(input.ph_cnr()) <= 0:
            ui.notification_show("Pick a campaign and enter a measured CNR greater than zero.", type="warning")
            return
        c.add_scan(float(input.ph_kv()), float(input.ph_mas()), float(input.ph_cnr()))
        c.save()
        phantom_version.set(phantom_version() + 1)
        ui.update_numeric("ph_cnr", value=0.0)

    @reactive.calc
    def phantom_state():
        c = campaign()
        if c is None:
            return None
        return c, suggest(c, dose_fn_from_predictor(pred()))

    @render.ui
    def ph_status():
        st = phantom_state()
        if st is None:
            return ui.div(ui.markdown("No campaign yet. Press **Start a campaign** to begin with your current "
                                      "protocol, or **Run a simulated demo** to see how it behaves."),
                          class_="verdict warn")
        c, r = st
        if r.next_scan:
            n = r.next_scan
            line = (f"**Scan the phantom at {n['kVp']:.0f} kV and {n['mAs']:.0f} mAs** "
                    f"(≈ {n['CTDIvol']:.1f} mGy), measure the CNR, and type it below.\n\n*{n['why']}*")
        else:
            line = f"**Campaign finished.** {r.reason}"
        rec_line = ""
        if r.recommended:
            rc = r.recommended
            rec_line = (f"\n\nBest protocol so far: **{rc['kVp']:.0f} kV / {rc['mAs']:.0f} mAs** "
                        f"— {rc['CTDIvol']:.1f} mGy, predicted CNR {rc['CNR_mean']:.2f}, "
                        f"{rc['PoF']*100:.0f}% chance of meeting the target.")
        g_line = ""
        if r.surrogate is not None:
            g_line = (f"\n\nMeasured iodine exponent γ = **{r.surrogate.gamma:.2f} ± "
                      f"{r.surrogate.gamma_sd:.2f}** — screen 2 now uses this instead of the textbook 1.35.")
        return ui.div(ui.markdown(f"{len(c.scans)} scan(s) recorded. " + line + rec_line + g_line),
                      class_="verdict ok")

    @render_widget
    def fig_phantom():
        st = phantom_state()
        req(st)
        c, r = st
        if r.surrogate is None or not len(r.grid):
            return fg.phantom_curve(pd.DataFrame(columns=["kVp", "CTDIvol", "CNR_mean", "CNR_lo", "CNR_hi"]),
                                    pd.DataFrame(columns=["kVp", "CTDI", "CNR"]), [], 1.0, None)
        f = dose_fn_from_predictor(pred())
        obs = c.scans_df()
        obs["CTDI"] = [row.CTDIvol if pd.notna(row.CTDIvol) else float(f(row.kVp, row.mAs))
                       for row in obs.itertuples()]
        return fg.phantom_curve(r.grid, obs, c.kv_options, r.target, r.recommended)

    # ------------------------------------------------------------------ advanced: bandit
    @reactive.effect
    @reactive.event(input.bd_propose)
    def _propose():
        b = bandit()
        p = b.propose(float(input.bmi()))
        last_proposal.set(p)

    @reactive.effect
    @reactive.event(input.bd_record)
    def _record():
        p = last_proposal()
        if p is None:
            ui.notification_show("Propose a protocol first.", type="warning")
            return
        score = input.bd_score()
        b = bandit()
        d = pred().predict_one(float(input.bmi()), p["kVp"], p["mAs"], float(input.length()))
        b.record(exam=input.exam(), BMI=float(input.bmi()), scan_length_cm=float(input.length()),
                 proposed_kVp=p["kVp"], proposed_mAs=p["mAs"], proposal_mode=p["mode"],
                 kVp=p["kVp"], mAs=p["mAs"], approved_as_proposed=True,
                 CTDIvol=d["CTDIvol"], DLP=d["DLP"], dose_source="predicted",
                 score=float(score) if score else None)
        b.save_log(input.exam())
        bandit_version.set(bandit_version() + 1)
        ui.notification_show("Recorded. The acceptability curve on screen 2 has been updated.", duration=5)

    @render.ui
    def bd_out():
        p = last_proposal()
        b = bandit()
        n = int(len(b.scored()))
        if p is None:
            return ui.div(ui.markdown(
                f"{n} scored examination(s) so far. Press **Propose a protocol** to get a suggestion for the "
                "patient currently in the sidebar. A radiologist approves it, the scan happens, and the score "
                "comes back here."), class_="verdict warn")
        d = pred().predict_one(float(input.bmi()), p["kVp"], p["mAs"], float(input.length()))
        return ui.div(ui.markdown(
            f"**Proposal: {p['kVp']:.0f} kV / {p['mAs']:.0f} mAs** — {d['CTDIvol']:.1f} mGy, "
            f"{fmt(d['DLP'])} mGy·cm, {p['saving_vs_ref_pct']:.0f}% below the local median for this BMI.\n\n"
            f"Chance of a diagnostic image: **{p['p_mean']*100:.0f}%** "
            f"(80% credible interval {p['p_lo']*100:.0f}–{p['p_hi']*100:.0f}%). Mode: {p['mode']}. "
            f"Learned from {n} scored examination(s)."), class_="verdict ok")

    @render.data_frame
    def tbl_bandit():
        bandit_version()
        log = DoseBandit.load_log(input.exam())
        if not len(log):
            return render.DataGrid(pd.DataFrame({"log": ["no examinations recorded yet"]}))
        out = pd.DataFrame({
            "When": pd.to_datetime(log.timestamp, errors="coerce").dt.strftime("%d %b %H:%M"),
            "BMI": pd.to_numeric(log.BMI, errors="coerce").round(1),
            "kV": pd.to_numeric(log.kVp, errors="coerce").round(0),
            "mAs": pd.to_numeric(log.mAs, errors="coerce").round(0),
            "Per slice (mGy)": pd.to_numeric(log.CTDIvol, errors="coerce").round(1),
            "Total (mGy·cm)": pd.to_numeric(log.DLP, errors="coerce").round(0),
            "Score": log.score, "Acceptable": log.acceptable})
        return render.DataGrid(out.tail(50), height="240px", width="100%")

    # ------------------------------------------------------------------ advanced: batch
    @reactive.calc
    def batch_table() -> pd.DataFrame | None:
        f = input.batch_file()
        if not f:
            return None
        df = load_table(f[0]["datapath"], default_exam=input.exam()).dropna(subset=FEATURES)
        out = pred().predict(df)
        for t in TARGETS:
            if t in df and df[t].notna().any():
                out[t + "_measured"] = df[t].to_numpy()
                out[t + "_error_%"] = (out[t] / out[t + "_measured"] - 1) * 100
        return out.round(2)

    @render.ui
    def batch_summary():
        out = batch_table()
        if out is None:
            return ui.markdown("Upload a file to predict every row in it.")
        lines = [f"**{len(out)} examinations** predicted."]
        for t in TARGETS:
            col = t + "_error_%"
            if col in out:
                lines.append(f"{t}: average error **{out[col].abs().mean():.2f}%**, "
                             f"median **{out[col].abs().median():.2f}%**")
        return ui.markdown("  \n".join(lines))

    @render.data_frame
    def tbl_batch():
        out = batch_table()
        if out is None:
            return render.DataGrid(pd.DataFrame({"file": ["none uploaded yet"]}))
        return render.DataGrid(out, height="320px", width="100%")

    @render.download(filename="ct_dose_predictions.csv")
    def batch_dl():
        out = batch_table()
        if out is not None:
            yield out.to_csv(index=False)


app = App(app_ui, server)


if __name__ == "__main__":
    import shiny
    print("\n  CT Dose AI — open http://127.0.0.1:8000 in your browser (Ctrl+C to stop)\n")
    shiny.run_app(app, host="127.0.0.1", port=8000, launch_browser=True)
