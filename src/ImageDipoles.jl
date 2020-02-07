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

@inline function plate_dipole_image(r::T,d::T,zplate) where T
    zrel = r[3] - zplate
    # position is reflected in z
    rp = T( r[1],  r[2], r[3]-2zrel)
    # dipole moment is reflected in x and y
    dp = T(-d[1], -d[2], d[3])
    return rp, dp
end

@inline function sphere_chargedipole_image(r,q,d,R)
    ar2 = sum(abs2,r)
    ar = sqrt(ar2)
    ar3 = ar2*ar
    rp = r * R^2/ar2
    rdotd = dot(r,d)
    dp = R^3/ar3 * (2r*rdotd/ar2 - d)
    qp = R*(rdotd/ar3 - q/ar)
    return rp, qp, dp
end
@inline function sphere_chargedipole_image(r,q,d,R,R0)
    rp, qp, dp = sphere_chargedipole_image(r-R0,q,d,R)
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
            push!.((rs, qs, ds), sphere_chargedipole_image(rs[end],qs[end],ds[end],Rs[ii%2+1],R0s[ii%2+1]))
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
mutable struct Twosphere_Chargedipole_ImageIter
    Rs::Tuple{Float64,Float64}
    R0s::Tuple{SV3,SV3}
    r::SV3
    q::Float64
    d::SV3
    nextsphere::Int
end
function Base.iterate(iter::Twosphere_Chargedipole_ImageIter,state=nothing)
    iter.r, iter.q, iter.d = sphere_chargedipole_image(iter.r,iter.q,iter.d,iter.Rs[iter.nextsphere],iter.R0s[iter.nextsphere])
    # 1 -> 2, 2 -> 1
    iter.nextsphere = 3 - iter.nextsphere
    return (iter.r, iter.q, iter.d), nothing
end

# stateful iterator over image dipoles of two parallel plates at zs
mutable struct Twoplate_Dipole_ImageIter
    zs::Tuple{Float64,Float64}
    r::SV3
    d::SV3
    nextplate::Int
end
function Base.iterate(iter::Twoplate_Dipole_ImageIter,state=nothing)
    iter.r, iter.d = plate_dipole_image(iter.r,iter.d,iter.zs[iter.nextplate])
    # 1 -> 2, 2 -> 1
    iter.nextplate = 3 - iter.nextplate
    return (iter.r, iter.d), nothing
end

function twoplate_V_dipole_dipole(zs,r1,d1,r2,d2,tol=1e-6)
    # interaction of dipole d1 at r1 with dipole r2 d2 and its images
    Vdd = V_dipole_dipole(r2,d2,r1,d1)
    Vdi = 0.
    Vmax = r1==r2 ? 0. : abs(Vdd)
    for iplate = 1:2
        ii = 0
        ismall = 0
        for (r,d) in Twoplate_Dipole_ImageIter(zs,r2,d2,iplate)
            ii += 1
            Vc = V_dipole_dipole(r,d,r1,d1)
            Vdi += Vc
            Vmax = max(Vmax,abs(Vc))
            ismall = abs(Vc)/abs(Vmax) < tol ? ismall+1 : 0
            ismall > 2 && break
        end
        #println("did isphere = $isphere, ii = $ii")
    end
    Vdd, Vdi
end

function twosphere_V_dipole_dipole(Rs,R0s,r1,d1,r2,d2,tol=1e-6)
    # interaction of dipole d1 at r1 with dipole r2 d2 and its images
    Vdd = V_dipole_dipole(r2,d2,r1,d1)
    Vdi = 0.
    Vmax = r1==r2 ? 0. : abs(Vdd)
    for isphere = 1:2
        ii = 0
        ismall = 0
        for (r,q,d) in Twosphere_Chargedipole_ImageIter(Rs,R0s,r2,0.,d2,isphere)
            ii += 1
            Vc = V_chargedipole_dipole(r,q,d,r1,d1)
            Vdi += Vc
            Vmax = max(Vmax,abs(Vc))
            ismall = abs(Vc)/abs(Vmax) < tol ? ismall+1 : 0
            ismall > 2 && break
        end
        #println("did isphere = $isphere, ii = $ii")
    end
    Vdd, Vdi
end

function environ_n_dipole_interactions(Vdidfun, rs, ds)
    N = length(rs)
    N == length(ds) || error("length(rs) = $(length(rs)) must be equal to length(ds) = $(length(ds))!")
    Vdd = Array{Float64}(undef,N,N)
    Vid = similar(Vdd)
    for ii = 1:length(rs)
        for jj = 1:ii
            Vdd[ii,jj], Vid[ii,jj] = Vdidfun(rs[ii],ds[ii],rs[jj],ds[jj])
            Vdd[jj,ii], Vid[jj,ii] = Vdd[ii,jj], Vid[ii,jj]
        end
        Vdd[ii,ii] = 0.
    end
    return Vdd,Vid
end

function twoplate_n_dipole_interactions(zs, rs, ds, tol=1e-6)
    Vdidfun(r1,d1,r2,d2) = twoplate_V_dipole_dipole(zs,r1,d1,r2,d2,tol)
    environ_n_dipole_interactions(Vdidfun, rs, ds)
end

function twosphere_n_dipole_interactions(Rs, R0s, rs, ds, tol=1e-6)
    Vdidfun(r1,d1,r2,d2) = twosphere_V_dipole_dipole(Rs,R0s,r1,d1,r2,d2,tol)
    environ_n_dipole_interactions(Vdidfun, rs, ds)
end

end # module
