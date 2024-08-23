using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
equations = CompressibleEulerEquations2D(1.4)

prandtl_number() = 0.72
mu() = 3.5e-5
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

Temp() = 300.0
p_inf() = 8610.0
l_inf() = 1.0
mach_inf() = 0.8
R() = 287.0
@inline function initial_condition_flatplate(x, t, equations)
    # set the freestream flow parameters
    gamma = equations.gamma

    v1 = 34.7189
    v2 = 0.0
    rho = p_inf()/(R() * Temp())

    prim = SVector(rho, v1, v2, p_inf())
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_flatplate

surface_flux = flux_lax_friedrichs

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

mesh_file = "plate_ray.inp"

mesh = P4estMesh{2}(mesh_file, initial_refinement_level = 0)

# Reynolds number at flat plate is 10^5.

# The boundary values across outer boundary are constant but subsonic, so we cannot compute the
# boundary flux from the external information alone. Thus, we use the numerical flux to distinguish
# between inflow and outflow characteristics
@inline function boundary_condition_subsonic_constant(u_inner,
                                                      normal_direction::AbstractVector, x,
                                                      t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach08_flow(x, t, equations)

    return Trixi.flux_hll(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = Dict(:PhysicalLine1 => boundary_condition_subsonic_constant,
                           :PhysicalLine2 => boundary_condition_subsonic_constant,
                           :PhysicalLine3 => boundary_condition_subsonic_constant,
                           :PhysicalLine4 => boundary_condition_subsonic_constant,
                           :PhysicalLine5 => boundary_condition_slip_wall,
                           :PhysicalLine6 => boundary_condition_slip_wall)

velocity_flatplate = NoSlip((x, t, equations) -> SVector(0.0, 0.0))

heat_flatplate = Adiabatic((x, t, equations) -> 0.0)

boundary_conditions_flatplate = BoundaryConditionNavierStokesWall(velocity_flatplate,
                                                                heat_flatplate)

velocity_bc_inlet = NoSlip((x, t, equations) -> (34.7189, 0.0))

# In the paper, the free stream pressure is specific while you are specifying the temperature (maybe)
heat_bc_square = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_square = BoundaryConditionNavierStokesWall(velocity_bc_square,
                                                              heat_bc_square)

boundary_conditions_parabolic = Dict(:PhysicalLine1 => boundary_condition_subsonic_constant,
                                     :PhysicalLine2 => boundary_condition_subsonic_constant,
                                     :PhysicalLine3 => boundary_condition_subsonic_constant,
                                     :PhysicalLine4 => boundary_condition_subsonic_constant,
                                     :PhysicalLine5 => boundary_condition_slip_wall,
                                     :PhysicalLine6 => boundary_condition_slip_wall) # TODO - There are only 5 lines

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers

# Run for a long time to reach a state where forces stabilize up to 3 digits
tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 2000

force_boundary_names = (:AirfoilBottom, :AirfoilTop)

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[])

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()); abstol = 1e-8,
            reltol = 1e-8,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary

