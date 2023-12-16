function [ CD ] = static_codegen_defs()
%STATIC_CODEGEN_DEFS Summary of this function goes here
%   Detailed explanation goes here

CD.paramTypeName = 'PL';
CD.defaultNcPfx = 'nc_';

%%
CD.fullNcPfx = [CD.paramTypeName '::' CD.defaultNcPfx];

end

