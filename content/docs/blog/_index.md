---
title: Differentiable Programming with Flux.jl
layout: post
mathjax: true
---

<!---
<head>
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
      },
      svg: {
        fontCache: 'global'
      }
    };
  </script>

  <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"/>
</head>
-->


Flux has taken some major strides in the past couple of years since it has been out. But its verstality can be subtle to grasp wothout actually using it. So this series is for bringing to notice how best to take advantage of Flux and its gradient-taking backend (automatic differentiation: AD for short) Zygote.

![flux-logo](https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/logo.png)

Starting with a bit of housekeeping. This piece will introduce some basic guidelines to Julia programming and should hopefully help with your understanding of the language and using it with a few neat tricks. Another task is to clarify what Flux and its ecosystem isn't. It **isn't a strictly deep learning library**, although it does have most of the primitives for deep learning defined. It is essentially a framework for differentiable programming.

For a TL;DR, differentiable programming ($\partial$P) is a way of treating arbitrary programs as differentiable. Put it easily, it is a generalisation of the way we treat deep learning as consisting of a forward pass and a backwards pass. It applies the chain rule (refer the equation below) to every operatoin that takes place in a program. The record and sequence of the operations to every element in the code is the code itself! Specifically, it is the AST of the code.

$$ \frac{ d{ (f \circ g)(x) }}{ d{x}} = \frac{\partial f}{\partial g} \times \frac{\partial g}{\partial x} $$

It replaces the standard neural network, with basically any other code and the model subtly melts away from being a sigular entity (think *Sequential* from Keras), to a part of general logic that you wish to implement. ~The adjoints are defined just as they are for any other differentiable function, again generalised from the mathematical priciples, to implementing logic consistent with what a deconstruction of the actions would look like.~

<!-- TODO: Add section on why adjoints are needed + block diagrams/ pictures; link to post about it -->

## How the adjoints are defined

Consider a regular function

```julia
function f(arg1::CustomType, arg2, arg3...)
  transform1 = f1(arg1)
  transform2 = f2(transform1, arg3...)
  result = f3(transform1, transform2)
  result
end
.
# going down in the abstraction
.

function f1(arg)
  result = operations_with_some_concrete_types(arg)
end
```

The function `f` can accept arguments with any type, including user defined ones. When we call on this function, it executes a bevy of other functions, ultimately ending with some basic operations involving concrete types (be they Arrays, Numbers, Symbols, etc), let's call them primitives. Let's now teach Julia how to differentiate operations involving these primitives. This would involve defining the adjoint for `sqrt` for a real number, for example.

$$ \frac{d{\sqrt{x}}}{d{x}} = \frac{1}{2\sqrt{x}} $$

Which can be expressed as

```julia
@adjoint Base.sqrt(x::Real) = Base.sqrt(x), Δ -> ( Δ * inv(2 * sqrt(x)),)
```

_The process and intuition to writing appropriate adjoints is a different blog._

If one could keep track of higher level operations, and define the adjoints on the primitives, we can essentially "solve back", accumulating the resulting gradients from all the transforms (with the help of the adjoints from the primitives), and maintaining some structures, like constructing `NamedTuple`s with the appropriate keys, we can express any operation as differentiable. The backwards pass flow would basically go something like **f3 --> f2 --> operations_with_some_concrete_types**. This way we can traverse our code (specifically, the [intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation)), and generate the backwards passes on the fly.

The cool part about this approach is that if we were to define the adjoints for the primitives or the base functions of a programming language, we can get any arbitrary program to be differentiated, and even support custom types and packages, almost for free. Add in an ideal optimising compiler, and these backwards passes become efficient too!

To give an example, the forward pass can be thought of as the process of tying your shoelaces, and the backwards pass is when we untie them by pulling the two ends apart.

<center>
	<image src="https://media2.giphy.com/media/12BVJ0EMPkepG0/giphy.gif", align="center">
</center>

For a lot of this to work as expected, though, it is pertinent that the base language on top of which this entire machinery is built, exposes meaningful expressions of its intermediate representation that can be used to infer the backwards passes on the fly, and this is precisely what Julia does, given its history of hackability. Flux takes this hackability, and runs with it to the point of making sure that the entire library is focussed on inviting people to its source code and in fact extending it with their own layers and definitions and optimisers and what have you. This is a tough ask, since it means anticipating which assumptions are safe, and which aren't, but it's defintely worth it, since it then allows users to gracefully add in complexity as required.

A post will be up later talking about implementing a differentiable programming solution and another explaining the guts of what makes Flux and Zygote tick.

## A Basic Optimisation Loop

For now, let's start with the classic example of optimising a random array to a different random array. It's just to illustrate how a simple iterative optimisation loop is expressed in Flux.

```julia
z = rand(3,3)
z′ = rand(3,3)

loss(x) = Flux.mse(z * x, z′ * x)
opt = Momentum()
ps = Params([z])  # z is an implicit parameter, and thus needs to be wrapped in the `Params` type.

for i = 1:10^5
  x = rand(3)
  gs = gradient(ps) do
    loss(x)
  end
  Flux.Optimise.update!(opt, ps, gs)
end

z ≈ z′ # true
```

And just like that, we have moved `z` close to `z′`!

## Adapting this to a custom type

Now, let's express this in terms of our own custom `struct`. For simplicity's sake, I am going to keep the fields of the `struct` Arrays, but they could be anything really.

```julia
import Base: +, -, *, /
import Base: isapprox
using MacroTools: @forward

mutable struct GG{T}
  a::T
  b::T
end

GG(a) = GG(a, a)

for op in (:+, :*, :-, :/)

  @eval @inline $(op)(a::GG, b::GG) = GG(broadcast($op, a.a, b.a), 
					broadcast($op, a.b, b.b))

  @eval @inline $(op)(a::GG, b) = GG(broadcast($op, a.a, b), 
				     broadcast($op, a.b, b))

  @eval @inline $(op)(b, a::GG) = GG(broadcast($op, a.a, b), 
				     broadcast($op, a.b, b))
end


@forward GG.a Base.size
```

Here, we've declared the struct, and defined some basic operations on how to handle the struct and its interaction with other types. Notice how we make use of Julia's excellent `broadcast`ing infrastructure, and a bit of code interpolation to avoid repeating defitintions for all the operations we want to hold it to, `(:+, :*, :-, :/)` in this case. `@inline` also hints to the Julia compiler that these operations can be inlined easily, and it should try to do this optimisation if possible.

And just to hit the nail on the head, let's define some more primitives that could come in handy while optimisation. These are operations that a lot of folks would already be used to doing for mathematical compute, but we will extrapolate it to arbitrary structs, that don't immediately make sense to be "optimisable", in a manner of speaking.

```julia
Base.zero(a::GG) = GG(zero(a.a), zero(a.b))
Base.length(::GG) = 1
Base.:^(a::GG, i) = GG(a.a .^ i, a.b .^ i)

import Statistics: mean

mean(a::GG) = mean(a.a) + mean(a.b)
Base.sum(a::GG) = sum(a.a) + sum(a.b)

Base.isapprox(a::GG{T}, b::GG{T}) where T = all([isapprox(a.a, b.a), isapprox(a.b, b.b)])
```

One last thing that might be necessary to take advantage of Flux's optimisers is to teach it what to do with the `GG` struct. We can extend it to just call `update` on all the fields of the struct.

```julia
function Flux.Optimise.update!(opt, x::T, gs, fs = fieldnames(T)) where {T<:GG}
  gs = gs.x
  for f in fs
    Flux.Optimise.update!(opt, getfield(x,f), getfield(gs,f))
  end
end
```

And with that, we should be ready to do our optimisation.

Let's define two instances of our `GG` struct that we'd like to optimise.

```julia
a = GG(rand(3,3), rand(3,3))
b = GG(rand(3,3), rand(3,3))
```

And we will use the same `Momentum` optimiser and mean-squared-error loss.

```julia
opt = Momentum()
for i = 1:10^5
  gs = gradient(a) do x
    sum((x-b) * (x-b)) / prod(size(x))
  end
  Flux.Optimise.update!(opt, a, gs[1])
end

a ≈ b # true
```

With this we have optimised a struct to another. Now we can take this concept and apply it to struct than a simple random array.

Another thing to note here is the complete lack of need of any call to `Params` in this case. This is because all of our parameters have been made explicit via passing `a`.

To give some context on the discussion earlier; the operations such as `sum`, `prod`, `size`, `-` etc are visible to Flux as valid operators to the parameters (`a`) and it looks into the implementation that we use for these transforms, to come up with valid `adjoint` methods. Think of it as the pulling motion from the shoelace example. Using these, it accumulates the gradients from all the operations, and finally returns them, keeping the structure of the paramters intact. This allows us to treat them as instances of the same type as usual, and finally optimise on them.

## Optimising Colours

With the example done, let's try optimising colours. This is going to get fun! This example is taken from some of our work in the differentiable programming examples that we present [here](https://github.com/MikeInnes/zygote-paper/blob/master/5_optimising_colours/run_colours.jl).

```julia
target = RGB(1, 0, 0)
colour = RGB(1, 1, 1)

function update_colour(c, Δ, η = 0.001)
   return RGB(
        c.r - η*Δ.r,
        c.g - η*Δ.g,
        c.b - η*Δ.b,
    )
end

for idx in 1:51
    global colour, target

    # Calculate gradients
    grads = Zygote.gradient(colour) do y
        colordiff(target, y)
    end
    # Update colour
    colour = update_colour(colour, grads[1])
    if idx % 5 == 1
        @info idx, colour
    end
end
```

Here our struct is just the `RGB` taken from the Colors.jl package. Again, the trick is to have meaningful operations defined on our type, based on the operations we will hit while calculating our loss function. The function `colordiff` already gives us the distance between two colours. It is important to note that the `Descent` optimiser does not check for convergence bounds and will ultimately diverge if the optimisation loop is not stopped.

I hope this helped motivate the different aspects of making a piece of code differentiable, and how that might be useful. The implementation need not be very complicated, if we understand the basic requirements for a library like Zygote. With the coming of [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) it should be possible to automate the optimisation over structs for many cases.

_Cheers_
