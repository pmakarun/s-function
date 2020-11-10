Simulink.importExternalCTypes('sfun_header.h');
def = legacy_code('initialize');
def.SFunctionName = 'osqp_test';
def.SourceFiles = {'sfun_source.c'};
def.HeaderFiles = {'sfun_header.h'};
def.OutputFcnSpec = ['void update_and_solve(StateSpace u1[1], ControlInputs y1[1])'];
legacy_code('sfcn_cmex_generate', def);
legacy_code('compile', def);
legacy_code('sfcn_tlc_generate', def);
legacy_code('rtwmakecfg_generate', def);
legacy_code('slblock_generate', def);
save('sfun_bus.mat','StateSpace','ControlInputs')