JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.activate("docs"); Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'
update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile(); Pkg.activate("docs"); Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

serve:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs()'

clean:
	rm -rf docs/build

.PHONY: init test serve clean
