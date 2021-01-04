### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 23ad7ed0-4ba2-11eb-1a12-13c1eb87a236
begin
	using DifferentialEquations
	using Plots
	using Parameters
	using StaticArrays
	using PlutoUI
end


# ╔═╡ 3ba55c10-4ba2-11eb-0bff-3f8ecd52ce2d
md"""
#### Using state machines and callbacks to model friction and limit stops

Modeling friction is always an annoying task, requiring extra characterization of
your system and convincing modeling tools to operate through all the nonlinearities
of the model.

Here we'll make a state machine to think about the different states of the model
and then use the [VectorContinuousCallback](https://diffeq.sciml.ai/stable/features/callback_functions/#DiffEqBase.VectorContinuousCallback) feature of
the Julia [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package.

Consider a sliding mass system in which an external force is applied to the
mass and the mass is subject to static and kinetic friction,
and bounded by limit stops which abruptly stop its motion.
"""

# ╔═╡ a6b8bc40-4c88-11eb-1b46-c727b7f022d7
Resource("https://klaff.github.io/friction_sm1.png")

# ╔═╡ ceaa0222-4ba4-11eb-0936-e9e1716a1338
md"""
We can identify five different states for the system:
- `LEFT_STOP`: mass against left stop, motion resisted by static friction and stop
- `SLIDING_RIGHT`: mass moving, being resisted by dynamic friction
- `STOPPED`: mass not against either stop, motion resisted by static friction
- `SLIDING_LEFT`: like `SLIDING_RIGHT` but in the other direction
- `RIGHT_STOP`: like `LEFT_STOP` but at the other end.
"""

# ╔═╡ 3d2c5c40-4c89-11eb-1524-bfe20b639b35
Resource("https://klaff.github.io/friction_sm2.png")

# ╔═╡ 2d67a5d0-4c89-11eb-0c44-b10b43eb1cdd
md"""
There are ten transitions between states, organized in eight lines:
- `LEFT_STOP` to `SLIDING_RIGHT` if force (on the mass) is greater than static friction
- `SLIDING_RIGHT` to `STOPPED` if speed drops to zero
- `SLIDING_RIGHT` to `RIGHT_STOP` or `SLIDING_LEFT` (depending on COR) if position reaches right stop
- `STOPPED` to `SLIDING_RIGHT` if force is greater than static friction
- `STOPPED` to `SLIDING_LEFT` if force is greater than static friction (but in the other direction)
- `SLIDING_LEFT` to `STOPPED` if speed drops to zero
- `SLIDING_LEFT` to `LEFT_STOP` or `SLIDING_RIGHT` if position reaches left stop
- `RIGHT_STOP` to `SLIDING_LEFT` if force is greater than static friction (but to the left)
"""

# ╔═╡ 4147d3a0-4ba7-11eb-0d3c-9990a1ea2554
md"""
Let's enumerate our states and make a default parameter object using some tools from [Parameters.jl](https://mauro3.github.io/Parameters.jl/stable/).
"""

# ╔═╡ 6b47a090-4ba7-11eb-053c-1dece557e20b
@enum State LEFT_STOP SLIDING_RIGHT STOPPED SLIDING_LEFT RIGHT_STOP


# ╔═╡ cae2e0a0-4ba7-11eb-1835-3f7fd824a8ce
@with_kw mutable struct Params
	M::Float64 = 1.0 # mass of sliding object in kg
	ga::Float64 = 9.81 # gravitational constant, m/s^2
	μs::Float64 = 0.8 # coefficient of static friction
	μk::Float64 = 0.5 # coefficient of kinematic friction
	cor::Float64 = 0.3 # coefficient of restitution, 0=inelastic, 1=elastic
	xls::Float64 = -0.5 # location of left stop in m
	xrs::Float64 = 0.5 # location of right stop in m
	state::State = STOPPED # state of system
end


# ╔═╡ 0953dfa0-4ba9-11eb-018c-9d3a0ae342fe
md"""
Let's write a state-aware differential equation.
"""

# ╔═╡ f31483a0-4baf-11eb-1a56-49cce578a7aa
md"""
The tests we use to move from one state to the other are defined in `FricSMCondx`.
"""

# ╔═╡ 3e942422-4dfe-11eb-2935-ff642f668365
md"""
The corresponding actions are defined in `FricSMaffect!`
"""

# ╔═╡ af3c5bc2-4c82-11eb-3058-d7a457ae6083
function FricSMaffect!(integ, idx)
	@unpack_Params integ.p
	print("affect,",integ.t,",",idx)
	if     idx==1 
		integ.p.state = SLIDING_RIGHT
	elseif idx==2
		integ.p.state = STOPPED
		integ.u[1] = 0.0
	elseif idx==3
		if cor==0.0
			integ.p.state = RIGHT_STOP
			integ.u[1] = 0.0
		else
			integ.p.state = SLIDING_LEFT
			integ.u[1] = -cor*integ.u[1]
		end
	elseif idx==4
		integ.p.state = SLIDING_RIGHT
	elseif idx==5
		integ.p.state = SLIDING_LEFT
	elseif idx==6 
		integ.p.state = STOPPED
		integ.u[1] = 0.0
	elseif idx==7
		if cor==0.0
			integ.p.state = LEFT_STOP
			integ.u[1] = 0.0
		else
			integ.p.state = SLIDING_RIGHT
			integ.u[1] = -cor*integ.u[1]
		end
	elseif idx==8 
		integ.p.state = SLIDING_LEFT
	end
end


# ╔═╡ 875db7a0-4bb2-11eb-2338-b5c75b1ff3c8
md"""
Create a forcing function.
"""

# ╔═╡ 98c2d7a0-4bb2-11eb-140b-6b8960fea14a
force_ext(t) = t<25 ? t/2*sin(2π*t/2) : (25-t/2)*sin(2π*t/2)


# ╔═╡ 1ab51d20-4bab-11eb-1533-618f186435a1
function sbode(u,p,t)
	v,x = u # unpacking state variables 
	@unpack_Params p # unpack parameters by name into function scope
	#netforce = calcnetforce(p, force_ext, t)
	if state==LEFT_STOP 
		dv = 0.0 # not moving
		dx = 0.0 # not moving
	elseif state==SLIDING_RIGHT
		dv = 1/M*(force_ext(t)-M*ga*μk)
		dx = v
	elseif state==STOPPED
		dv = 0.0
		dx = 0.0
	elseif state==SLIDING_LEFT
		dv = 1/M*(force_ext(t)+M*ga*μk)
		dx = v
	else # state==RIGHT_STOP
		dv = 0.0
		dx = 0.0
	end
	@SVector[dv, dx]
end
		

# ╔═╡ a2517940-4c82-11eb-3f2c-fb29882d4347
function FricSMCondx(out, u, t, integ)
	@unpack_Params integ.p
	v,x = u
	out[1] = (state==LEFT_STOP)*(force_ext(t)-M*ga*μs)
	out[2] = (state==SLIDING_RIGHT)*(-v)
	out[3] = (state==SLIDING_RIGHT)*(x-xrs)
	out[4] = (state==STOPPED)*(force_ext(t)-M*ga*μs)
	out[5] = (state==STOPPED)*(-force_ext(t)-M*ga*μs)
	out[6] = (state==SLIDING_LEFT)*v
	out[7] = (state==SLIDING_LEFT)*(-(x-xls))
	out[8] = (state==RIGHT_STOP)*(-(force_ext(t)+M*ga*μs))
end


# ╔═╡ ccc7d3a0-4baf-11eb-1a77-4ddeac196b3b
cb = VectorContinuousCallback(FricSMCondx, FricSMaffect!, nothing, 8);
		

# ╔═╡ c4e47320-4bb2-11eb-1fa1-1fcbf15af02b
md"""
Set up a sim function which sets initial conditions, timespan, parameters, and
simulation options. We set `dtmax = 0.01` because without it the solver
skips forward too quickly.
"""

# ╔═╡ cf2c4920-4bb2-11eb-117b-dd100b07babc
function sim()
	u0 = [0.0, 0.0]
	tspan = (0.0, 50.0)
	p = Params()
	prob = ODEProblem(sbode, u0, tspan, p)
	sol = solve(prob, callback = cb, dtmax=0.01)
end


# ╔═╡ 0baa2022-4bb3-11eb-1d81-0d20f28c073b
sol = sim();


# ╔═╡ 6668ee50-4c8b-11eb-3ac0-5575ff0df653
md"""
Plot the results! Here we can see the system being pushed back and forth by
an amplitude-modulated sinusoidal force.
"""

# ╔═╡ 1a415310-4bb3-11eb-210a-fdd4cc972288
let
	p1 = plot(sol, vars=[1], ylabel = "velocity, m/s")
	p2 = plot(sol, vars=[2], ylabel = "position, m")
	p0 = plot(sol.t, force_ext, ylabel = "force, N")
	plot(p0,p1,p2,layout=(3,1), link=:x, leg = false)
end


# ╔═╡ 34cadbb0-4df8-11eb-0e2e-4188b46d0ced
md"""
##### Conclusions
The [callback](https://diffeq.sciml.ai/stable/features/callback_functions/)
feature of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package
allows one to implement a state machine to change the behavior of a system
by letting the user
- Define a set of states,
- Define the behavior of each state in the ODE function,
- Define the conditions on which states will change in a conditions function, and
- Define what changes occur to the simulation on each change in the affect function.

The user can in this way define highly nonlinear actions and systems with
discontinuous derivatives without causing solver issues because the solvers are
aware of the callbacks and do not try to solve across the callback points.
"""

# ╔═╡ cdb08a22-4ba8-11eb-03b4-73109bae35d8
md"""
This [Julia language](https://julialang.org/) document was prepared using the [Pluto reactive notebook system](https://github.com/fonsp/Pluto.jl).

David Klaffenbach, 2021-01-01.

Revised 2021-01-03.
"""

# ╔═╡ Cell order:
# ╟─3ba55c10-4ba2-11eb-0bff-3f8ecd52ce2d
# ╟─a6b8bc40-4c88-11eb-1b46-c727b7f022d7
# ╟─ceaa0222-4ba4-11eb-0936-e9e1716a1338
# ╟─3d2c5c40-4c89-11eb-1524-bfe20b639b35
# ╟─2d67a5d0-4c89-11eb-0c44-b10b43eb1cdd
# ╟─4147d3a0-4ba7-11eb-0d3c-9990a1ea2554
# ╠═6b47a090-4ba7-11eb-053c-1dece557e20b
# ╠═cae2e0a0-4ba7-11eb-1835-3f7fd824a8ce
# ╟─0953dfa0-4ba9-11eb-018c-9d3a0ae342fe
# ╠═1ab51d20-4bab-11eb-1533-618f186435a1
# ╟─f31483a0-4baf-11eb-1a56-49cce578a7aa
# ╠═a2517940-4c82-11eb-3f2c-fb29882d4347
# ╟─3e942422-4dfe-11eb-2935-ff642f668365
# ╠═af3c5bc2-4c82-11eb-3058-d7a457ae6083
# ╠═ccc7d3a0-4baf-11eb-1a77-4ddeac196b3b
# ╟─875db7a0-4bb2-11eb-2338-b5c75b1ff3c8
# ╠═98c2d7a0-4bb2-11eb-140b-6b8960fea14a
# ╟─c4e47320-4bb2-11eb-1fa1-1fcbf15af02b
# ╠═cf2c4920-4bb2-11eb-117b-dd100b07babc
# ╠═0baa2022-4bb3-11eb-1d81-0d20f28c073b
# ╟─6668ee50-4c8b-11eb-3ac0-5575ff0df653
# ╠═1a415310-4bb3-11eb-210a-fdd4cc972288
# ╠═23ad7ed0-4ba2-11eb-1a12-13c1eb87a236
# ╟─34cadbb0-4df8-11eb-0e2e-4188b46d0ced
# ╟─cdb08a22-4ba8-11eb-03b4-73109bae35d8
