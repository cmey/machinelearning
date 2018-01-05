# Pkg.add("Knet")
using Knet


# Linear regression

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = mean(abs2,y-predict(w,x))

lossgradient = grad(loss)

function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
    	for i in 1:length(w)
    	    w[i] -= lr * dw[i]
    	end
    end
    return w
end


# example on Boston Housing data
include(Knet.dir("data","housing.jl"))
x,y = housing()
w = Any[ 0.1*randn(1,13), 0.0 ]

for i=1:10; train(w, [(x,y)]); println(loss(w,x,y)); end



# Softmax classification

predict(w,x) = w[1]*mat(x) .+ w[2]

loss(w,x,ygold) = nll(predict(w,x), ygold)  # neg log likelihood

lossgradient = grad(loss)

# example on MNIST

include(Knet.dir("data","mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)
w = Any[ 0.1f0*randn(Float32,10,784), zeros(Float32,10,1) ]
println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
for epoch=1:10
    train(w, dtrn; lr=0.5)
    println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end



# Multi-layer perceptron

function predict(w,x)
    x = mat(x)
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)  # neg log likelihood

lossgradient = grad(loss)

# single hidden layer of 64 units:
w = Any[ 0.1f0*randn(Float32,64,784), zeros(Float32,64,1),
         0.1f0*randn(Float32,10,64),  zeros(Float32,10,1) ]

function train(model, data, optim)
    for (x,y) in data
        grads = lossgradient(model,x,y)
        update!(model, grads, optim)
    end
end

o = optimizers(w, Adam)
println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
for epoch=1:10
    train(w, dtrn, o)
    println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end


# Convolutional neural network

function predict(w,x0)
    x1 = pool(relu.(conv4(w[1],x0) .+ w[2]))
    x2 = pool(relu.(conv4(w[3],x1) .+ w[4]))
    x3 = relu.(w[5]*mat(x2) .+ w[6])
    return w[7]*x3 .+ w[8]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)  # neg log likelihood

lossgradient = grad(loss)

w = Any[ xavier(Float32,5,5,1,20),  zeros(Float32,1,1,20,1),
         xavier(Float32,5,5,20,50), zeros(Float32,1,1,50,1),
         xavier(Float32,500,800),   zeros(Float32,500,1),
         xavier(Float32,10,500),    zeros(Float32,10,1) ]

# Use GPU arrays: Nope, getting "bad device id -1".
# dtrn = minibatch(xtrn,ytrn,100,xtype=KnetArray)
# dtst = minibatch(xtst,ytst,100,xtype=KnetArray)
# w = map(KnetArray, w)
dtrn = minibatch(xtrn,ytrn,100)
dtst = minibatch(xtst,ytst,100)

function train(model, data, optim)
    for (x,y) in data
        grads = lossgradient(model,x,y)
        update!(model, grads, optim)
    end
end

println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
for epoch=1:10
    train(w, dtrn, o)
    println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end



# Recurrent neural network
include(Knet.dir("data","gutenberg.jl"))
trn,tst,chars = shakespeare()  # TODO: 404, http://www.gutenberg.org/files/100/100-0.txt
map(summary,(trn,tst,chars))
println(string(chars[trn[1020:1210]]...))

# Minibatch into (256,100) blocks

BATCHSIZE = 256  # number of sequences per minibatch
SEQLENGTH = 100  # sequence length for bptt

function mb(a)
    N = div(length(a),BATCHSIZE)
    x = reshape(a[1:N*BATCHSIZE],N,BATCHSIZE)' # reshape full data to (B,N) with contiguous rows
    minibatch(x[:,1:N-1], x[:,2:N], SEQLENGTH) # split into (B,T) blocks
end

dtrn,dtst = mb(trn),mb(tst)
map(length, (dtrn,dtst))
# (192, 20)


RNNTYPE = :lstm  # can be :lstm, :gru, :tanh, :relu
NUMLAYERS = 1    # number of RNN layers
INPUTSIZE = 168  # size of the input character embedding
HIDDENSIZE = 334 # size of the hidden layers
VOCABSIZE = 84   # number of unique characters in data

function initmodel()
    w(d...)=KnetArray(xavier(Float32,d...))
    b(d...)=KnetArray(zeros(Float32,d...))
    r,wr = rnninit(INPUTSIZE,HIDDENSIZE,rnnType=RNNTYPE,numLayers=NUMLAYERS)
    wx = w(INPUTSIZE,VOCABSIZE)
    wy = w(VOCABSIZE,HIDDENSIZE)
    by = b(VOCABSIZE,1)
    return r,wr,wx,wy,by
end

# The predict function below takes:
# - weights ws,
# - inputs xs,
# - the initial hidden and cell states hx and cx
# - and returns output scores ys and the final hidden and cell states hy and cy
function predict(ws,xs,hx,cx)
    r,wr,wx,wy,by = ws
    x = wx[:,xs]                                         # xs=(B,T) x=(X,B,T)
    y,hy,cy = rnnforw(r,wr,x,hx,cx,hy=true,cy=true)      # y=(H,B,T) hy=cy=(H,B,L)
    ys = by.+wy*reshape(y,size(y,1),size(y,2)*size(y,3)) # ys=(H,B*T)
    return ys, hy, cy
end


function loss(w,x,y,h)
    py,hy,cy = predict(w,x,h...)
    h[1],h[2] = getval(hy),getval(cy)
    return nll(py,y)
end

lossgradient = gradloss(loss)

function train(model,data,optim)
    hiddens = Any[nothing,nothing]
    losses = []
    for (x,y) in data
        grads,loss1 = lossgradient(model,x,y,hiddens)
        update!(model, grads, optim)
	push!(losses, loss1)
    end
    return mean(losses)
end

function test(model,data)
    hiddens = Any[nothing,nothing]
    losses = []
    for (x,y) in data
        push!(losses, loss(model,x,y,hiddens))
    end
    return mean(losses)
end

# TAKES 10 minutes on a K80 !
EPOCHS = 30
model = initmodel()
optim = optimizers(model, Adam)
@time for epoch in 1:EPOCHS
    @time trnloss = train(model,dtrn,optim) # ~18 seconds
    @time tstloss = test(model,dtst)        # ~0.5 seconds
    println((:epoch, epoch, :trnppl, exp(trnloss), :tstppl, exp(tstloss)))
end


function generate(model,n)
    function sample(y)
        p,r=Array(exp.(y-logsumexp(y))),rand()
        for j=1:length(p); (r -= p[j]) < 0 && return j; end
    end
    h,c = nothing,nothing
    x = findfirst(chars,'\n')
    for i=1:n
        y,h,c = predict(model,[x],h,c)
        x = sample(y)
        print(chars[x])
    end
    println()
end

generate(model,1000)
