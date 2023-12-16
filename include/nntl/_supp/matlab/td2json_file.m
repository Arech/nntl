function td2json_file( td, fname, bTranspose )
%TD2JSON_FILE serializes td var into file fname
% uses https://github.com/christianpanton/matlab-json/

bTranspose = exist('bTranspose','var') && logical(bTranspose);

if ~isfield(td,'bColMajor') || ~isscalar(td.bColMajor) || ~islogical(td.bColMajor)
	td.bColMajor=true;
end

[fid,err]=fopen(fname, 'w');
if -1==fid
	disp(['Failed to open file: ' err]);
else
	if bTranspose
		td = struct_transpose(td);
	end
	
	fprintf(fid,tojson(td));
	fclose(fid);
end


end

