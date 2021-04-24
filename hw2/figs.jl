### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 0aaf4b08-a4a5-11eb-0732-0936be238136
using PyPlot

# ╔═╡ 296e89fa-7aca-4368-a5a1-526383e45f7e
timing_results = transpose([0.6685118675 0.3330111504 11.35976481 4.745308161; 19.74084425 2.095619678 99.41173959 11.46811938; 416.4413614 17.58693027 NaN 70.98236918; NaN 129.7298446 NaN 333.6011386])

# ╔═╡ ac564bd6-2fd3-4dcf-8e77-de451cd3ee74
rows = ["Serial v1", "Serial v2", "Distrib v1", "Distrib v2"]

# ╔═╡ bd6db598-6e66-4f02-9d5a-566efb696686
size = [8, 16, 32, 50]

# ╔═╡ 67b2bc54-dc7d-4b3a-8605-d1377b6c35f9
colors = ["red", "green", "blue", "black"]

# ╔═╡ 1bbe8953-0387-49bd-9ae2-6b78021a7b0a
begin
	pygui(false)
	figure()
	title("Timing Results")
	xlabel("State-Space Size")
	ylabel("Runtime (s)")
	for i ∈ 1:length(size)
	    plot(size, timing_results[i,:], label=rows[i], c=colors[i])
	end
	legend()
	gcf()
end

# ╔═╡ 39700669-4670-4eae-85b1-8ee47b6c5995
timing2 = transpose([25.45262504 1.873035431; 11.35976481 4.745308161; 6.147414923 2.783768415])

# ╔═╡ 117ee570-fabf-4b8c-8127-c143c9a54861
workers = [2, 4, 8]

# ╔═╡ b9b334cb-8579-4319-a2f7-22b427067da3
rows2 = ["Distrib v1", "Distrib v2"]

# ╔═╡ 88de8f80-ad16-42a8-8752-64dba595c8ae
begin
	pygui(false)
	figure()
	title("Timing Results")
	xlabel("State-Space Size")
	ylabel("Runtime (s)")
	plot(workers, timing2[1,:], label=rows2[1], c=colors[3])
	plot(workers, timing2[2,:], label=rows2[2], c=colors[4])
	legend()
	gcf()
end

# ╔═╡ Cell order:
# ╠═0aaf4b08-a4a5-11eb-0732-0936be238136
# ╠═296e89fa-7aca-4368-a5a1-526383e45f7e
# ╠═ac564bd6-2fd3-4dcf-8e77-de451cd3ee74
# ╠═bd6db598-6e66-4f02-9d5a-566efb696686
# ╠═67b2bc54-dc7d-4b3a-8605-d1377b6c35f9
# ╠═1bbe8953-0387-49bd-9ae2-6b78021a7b0a
# ╠═39700669-4670-4eae-85b1-8ee47b6c5995
# ╠═117ee570-fabf-4b8c-8127-c143c9a54861
# ╠═b9b334cb-8579-4319-a2f7-22b427067da3
# ╠═88de8f80-ad16-42a8-8752-64dba595c8ae
