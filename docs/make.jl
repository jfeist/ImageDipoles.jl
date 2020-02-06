using Documenter, ImageDipoles

makedocs(;
    modules=[ImageDipoles],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/jfeist/ImageDipoles.jl/blob/{commit}{path}#L{line}",
    sitename="ImageDipoles.jl",
    authors="Johannes Feist",
    assets=String[],
)

deploydocs(;
    repo="github.com/jfeist/ImageDipoles.jl",
)
