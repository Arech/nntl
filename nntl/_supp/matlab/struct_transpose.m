function [ s ] = struct_transpose( s )
%STRUCT_TRANSPOSE transposes all struct s member matrices

assert(isstruct(s));

fn = fieldnames(s);
for ii=1:length(fn)
	if (ismatrix(s.(fn{ii})))
		s.(fn{ii}) = transpose (s.(fn{ii}));
	end
end

if isfield(s,'bColMajor')
	s.bColMajor = ~s.bColMajor;
end

