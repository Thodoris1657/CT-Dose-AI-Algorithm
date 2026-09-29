@echo off
REM Builds both PDFs (run pdflatex twice each so references settle)
cd /d "%~dp0"
pdflatex -interaction=nonstopmode CT_Dose_AI_Report.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Report.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Presentation.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Presentation.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Cheatsheet.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Cheatsheet.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Advanced_Track.tex
pdflatex -interaction=nonstopmode CT_Dose_AI_Advanced_Track.tex
echo.
echo Done: CT_Dose_AI_Report.pdf and CT_Dose_AI_Presentation.pdf
pause
