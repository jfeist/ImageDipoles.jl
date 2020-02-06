module ImageDipoles

using LinearAlgebra
using StaticArrays

export SV3, dipole_image_interaction

const SV3 = SVector{3,Float64};

"Calculates the electric potential at point r of a charge q₁ and permanent dipole d₁ at position r₁.
   Distances are measured in nm, charge in units of e, potential in V.
   The full expression is:
     ϕ = (q₁ + d₁⋅rᵣ/Rᵣ²) / (4πϵ₀ Rᵣ)
   where rᵣ = r₁-r and Rᵣ = |rᵣ|
   In our units, 1/(4πϵ₀) = 1.4399644149416115 V nm/e"
@inline function ϕ_chargedipole(r₁,q₁,d₁,r)
    rᵣ = r - r₁
    Rᵣ2 = sum(abs2,rᵣ) + 1e-200
    ϕRᵣ = q₁ + d₁⋅rᵣ/Rᵣ2
    return 1.4399644149416115 * ϕRᵣ / √Rᵣ2
end

"Calculates the electric field at point r of a charge q₁ and permanent dipole d₁ at position r₁.
   Distances are measured in nm, charge in units of e, potential in V.
   The full expression is:
     E = ((q₁ + 3d₁⋅rᵣ/Rᵣ²) rᵣ - d₁) / (4πϵ₀ Rᵣ³)
   where rᵣ = r₁-r and Rᵣ = |rᵣ|
   In our units, 1/(4πϵ₀) = 1.4399644149416115 V nm/e"
@inline function E_chargedipole(r₁,q₁,d₁,r)
    rᵣ = r - r₁
    Rᵣ2 = sum(abs2,rᵣ) + 1e-200
    ERᵣ3 = (q₁ + 3d₁⋅rᵣ/Rᵣ2) * rᵣ - d₁
    return 1.4399644149416115 * ERᵣ3 / (Rᵣ2*√Rᵣ2)
end

@inline E_dipole(r1,d1,r) = E_chargedipole(r1,0.,d1,r)
@inline V_chargedipole_dipole(r1,q1,d1,r2,d2) = -dot(E_chargedipole(r1,q1,d1,r2),d2);
@inline V_dipole_dipole(r1,d1,r2,d2) = -dot(E_dipole(r1,d1,r2),d2);

function V_dipoles_images(rs, ds, rims, qims, dims)
    N = length(rs)
    M = length(rims)
    length(rs)==length(ds)==N || error("Must have length(rs)==length(ds)==N!")
    length(rims)==length(dims)==length(qims)==M || error("Must have length(rims)==length(dims)==length(qims)==M!")
    # array of dipole-dipole interactions
    Vdd = zeros(N,N)
    # array of dipole-image interactions (include factor 1/2)
    Vid = zeros(M,N)
    @views for ii = 1:N
        r1 = rs[ii]
        d1 = ds[ii]
        for jj = ii+1:N
            Vdd[jj,ii] = V_dipole_dipole(r1,d1,rs[jj],ds[jj])
            Vdd[ii,jj] = Vdd[jj,ii]
        end
        for jj = 1:M
            Vid[jj,ii] = 0.5 * V_chargedipole_dipole(rims[jj],qims[jj],dims[jj],r1,d1)
        end
    end
    return Vdd, Vid
end

V_dipoles(rs,ds) = V_dipoles_images(rs, ds, (), (), ())[1];

mutable struct ChargesDipoles
    rs::Vector{SV3}
    qs::Vector{Float64}
    ds::Vector{SV3}
end
ϕfun(cd::ChargesDipoles) = r -> sum(ϕ_chargedipole.(cd.rs,cd.qs,cd.ds,(r,)));
Efun(cd::ChargesDipoles) = r -> sum(E_chargedipole.(cd.rs,cd.qs,cd.ds,(r,)));
ϕfun(cd1::ChargesDipoles,cd2::ChargesDipoles) = r -> sum(ϕ_chargedipole.(cd1.rs,cd1.qs,cd1.ds,(r,))) + sum(ϕ_chargedipole.(cd2.rs,cd2.qs,cd2.ds,(r,)));
Efun(cd1::ChargesDipoles,cd2::ChargesDipoles) = r -> sum(E_chargedipole.(cd1.rs,cd1.qs,cd1.ds,(r,))) + sum(E_chargedipole.(cd2.rs,cd2.qs,cd2.ds,(r,)));

function sphere_chargedipole_images(r,q,d,R)
    ar2 = sum(abs2,r)
    ar = sqrt(ar2)
    ar3 = ar2*ar
    rp = r * R^2/ar2
    rdotd = dot(r,d)
    dp = R^3/ar3 * (2r*rdotd/ar2 - d)
    qp = R*(rdotd/ar3 - q/ar)
    return rp, qp, dp
end
function sphere_chargedipole_images(r,q,d,R,R0)
    rp, qp, dp = sphere_chargedipole_images(r-R0,q,d,R)
    return rp + R0, qp, dp
end

function twosphere_chargedipole_images(r0,q0,d0,Rs,R0s,tol=1e-6)
    rrs, qqs, dds = [r0], [q0], [d0]
    tol = tol*(norm(d0)+abs(q0))
    @views for ii=0:1
        # do not get images starting in sphere if initial position is inside sphere
        norm(r0-R0s[ii%2+1])<=Rs[ii%2+1] && continue
        rs, qs, ds = [r0], [q0], [d0]
        while norm(ds[end])>tol || abs(qs[end])>tol
            push!.((rs, qs, ds), sphere_chargedipole_images(rs[end],qs[end],ds[end],Rs[ii%2+1],R0s[ii%2+1]))
            ii += 1
        end
        # take 2:end-1 because the first one is the original dipole and the last one is below the cutoff
        rrs = [rrs;rs[2:end-1]]
        qqs = [qqs;qs[2:end-1]]
        dds = [dds;ds[2:end-1]]
    end
    return ChargesDipoles(rrs,qqs,dds)
end

# stateful iterator over image chargedipoles
mutable struct Twosphere_Chargedipole_Images
    Rs::Tuple{Float64,Float64}
    R0s::Tuple{SV3,SV3}
    r::SV3
    q::Float64
    d::SV3
    nextsphere::Int
end

function Base.iterate(iter::Twosphere_Chargedipole_Images,state=nothing)
    iter.r, iter.q, iter.d = sphere_chargedipole_images(iter.r,iter.q,iter.d,iter.Rs[iter.nextsphere],iter.R0s[iter.nextsphere])
    # 1 -> 2, 2 -> 1
    iter.nextsphere = 3 - iter.nextsphere
    return (iter.r, iter.q, iter.d), nothing
end

function dipole_image_interaction(Rs,R0s,r0,d0,tol=1e-7)
    isfirst = false
    iter = Twosphere_Chargedipole_Images(Rs,R0s,r0,0.,d0,1)
    (r,q,d), state = iterate(iter)
    V0 = V_chargedipole_dipole(r,q,d,r0,d0)
    V = V0
    ii = 0
    ismall = 0
    for (r,q,d) in iter
        ii += 1
        Vc = V_chargedipole_dipole(r,q,d,r0,d0)
        V += Vc
        ismall = abs(Vc)/abs(V) < tol ? ismall+1 : 0
        ismall > 3 && break
    end
    V, ii
end

function n_dipole_interactions(rs, ds, Rs, R0s, tol=1e-6)
    myimgs(r,d) = twosphere_chargedipole_images(r,0.,d,Rs,R0s,tol)
    cdims = myimgs.(rs,ds)
    # length(ri)-1 because the first element is the original dipole
    imgsizes = [length(cdi.rs)-1 for cdi in cdims]
    imstarts = cumsum([1;imgsizes])
    # all images in a row (imstarts[i]:imstarts[i+1] should give images of dipole i)
    rims = Vector{SV3}(undef,imstarts[end]-1)
    qims = similar(rims,Float64)
    dims = similar(rims)
    for ii=1:length(cdims)
        # cdims 2:end only (without original dipole)
        rims[imstarts[ii]:imstarts[ii+1]-1] = cdims[ii].rs[2:end]
        qims[imstarts[ii]:imstarts[ii+1]-1] = cdims[ii].qs[2:end]
        dims[imstarts[ii]:imstarts[ii+1]-1] = cdims[ii].ds[2:end]
    end
    cdim = ChargesDipoles(rims,qims,dims)
    pprintln("length(rims) = $(length(rims))")
    Vdd, Vid = V_dipoles_images(rs,ds,cdim.rs,cdim.qs,cdim.ds)
    Vid_r = similar(Vdd)
    for ii = 1:length(rs)
        Vid_r[ii,:] .= dropdims(sum(Vid[imstarts[ii]:imstarts[ii+1]-1,:],dims=1),dims=1)
    end
    cd = ChargesDipoles(rs,zeros(length(rs)),ds)
    return (Rs=Rs, R0s=R0s, cd=cd, cdim=cdim, Vdd=Vdd, Vid=Vid, Vid_r=Vid_r, imstarts=imstarts)
end

end # module
