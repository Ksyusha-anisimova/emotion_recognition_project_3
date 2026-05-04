# Design: Conclusion and Sources Docx

## Goal
Create a new Word document `заключение_и_источники.docx` (do not modify existing `диплом.docx`) containing two sections: **ЗАКЛЮЧЕНИЕ** and **СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ**. The style must match the main report (Times New Roman 14 pt, 1.5 spacing, justified, first-line indent 1.25 cm).

## Constraints
- Use the existing report styling as the template.
- Conclusion length: 1.5–2 pages.
- Sources: 25–30 items, GOST 7.0.5–2008 format, include URL and access date.
- Dates distributed over ~6 months before 18.03.2026, logically by project phase.
- RU/EN ratio ~60/40.
- Sources must be real and accessible (verified via web). 
- No edits to `диплом.docx`.

## Approach
1. **Template-based docx assembly**: unpack `диплом.docx`, reuse its paragraph formatting (heading + body + list) for consistent style, then build a new `document.xml` with just the two sections and repack into a new file.
2. **Content preparation**: write a full conclusion based strictly on project facts (3 classes, 48x48, EmotionCNN, local web app, reported accuracy 74.67%, etc.).
3. **Bibliography**: compile 26 sources (16 RU, 10 EN), format per GOST, add URL + access date.

## Data Flow
- Extract heading/body/list paragraph properties from `диплом.docx`.
- Generate paragraphs in WordprocessingML for:
  - Heading: "ЗАКЛЮЧЕНИЕ"
  - Conclusion text paragraphs
  - Heading: "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ"
  - Numbered list of sources
- Replace the body in a copied docx package and validate.

## Risks
- `docx-js` cannot be installed (network blocked), so we must edit XML directly.
- GOST formatting details may vary; use a conservative, widely accepted electronic-resource format.

## Validation
- Ensure doc opens without errors in Word/LibreOffice.
- Check formatting matches the original report (size, spacing, indent, justification).
- Confirm source count and RU/EN ratio.

## Notes
- Git repo not present, so no commit will be created.
