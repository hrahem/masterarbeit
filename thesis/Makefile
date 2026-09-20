DOC=internship_report

all:
	pdflatex $(DOC)
	bibtex $(DOC)
	pdflatex $(DOC)
	pdflatex $(DOC)
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $(DOC).log) \
	do pdflatex $(DOC); done	
clean:
	find -name '*~' -exec rm -f {} \;
	rm -rf *.log *.ps *.toc *.lo* *.bbl *.blg *.dvi *.idx *.ind *.ist *.glo *.glg *.alg *.acn *.acr *.out *.ilg *.gls *.aux *.pdf *.xml *.tdo *-blx.bib chapters/*.aux
