### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 0b95fb70-98cc-11eb-2809-1b796ba93bd4
using PyPlot

# ╔═╡ 58bb242c-1297-4d5c-8ed6-4a83d420ed60
begin
	n = 16 # used between 1 and 16 CPUs
	reduce_times = zeros(n)
	total_times = zeros(n)
	
	for i ∈ 1:n
		filetext = String.(eachline(joinpath(pwd(), "output/output_$i.out")))
		reduce_times[i] = parse(Float64, split(filetext[6], " ")[3])
		total_times[i] = parse(Float64, split(filetext[7], " ")[3])
	end
end

# ╔═╡ ec1663c9-cdcc-4246-9230-910fad7bff33
begin
	figure()
	title("Map-Reduce Parallel Execution")
	xlabel("Number of CPUs")
	ylabel("Time (s)")
	scatter(1:n, reduce_times, label="reduce")
	scatter(1:n, total_times, label="total")
	legend()
	gcf()
end

# ╔═╡ Cell order:
# ╠═0b95fb70-98cc-11eb-2809-1b796ba93bd4
# ╠═58bb242c-1297-4d5c-8ed6-4a83d420ed60
# ╠═ec1663c9-cdcc-4246-9230-910fad7bff33
