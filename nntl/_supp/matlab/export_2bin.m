function export_2bin( S, fname )
%EXPORT_STRUCT_2BIN Export 2D matrix or struct to NNTL binary file
% (see nntl/_supp/binfile.h for specifications)

MAX_FIELD_NAME_LENGTH=15;

if ~isstruct(S) && ismatrix(S)
	S=struct('mtx',S);
end
if ~isstruct(S)
	error('S must be a 2D matrix or struct');
end

[fid,err]=fopen(fname,'w','l');
if -1==fid
	error('Failed to open file %s: %s\n',fname,err);
end

fn = fieldnames(S);
fc = length(fn);

%bin_file::HEADER
fwrite(fid,'nntl','char');
fwrite(fid,fc,'uint16');

%bin_file::FIELD_ENTRY
for fidx=1:fc
	fld = S.(fn{fidx});
	if ~(isscalar(fld) || isvector(fld) || ismatrix(fld))
		fclose(fid);
		error('Unsupported field %s type', fn{fidx});
	end
	fldnameLenRes = MAX_FIELD_NAME_LENGTH - length(fn{fidx});
	if fldnameLenRes<0
		fclose(fid);
		error('Too long field name (%d or less is acceptable): %s',MAX_FIELD_NAME_LENGTH,fn{fidx});
	end
	
	[nrows,ncols]=size(fld);
	fwrite(fid,nrows,'uint32');
	fwrite(fid,ncols,'uint32');
	
	%fwrite(fid,sprintf( ['%-' num2str(MAX_FIELD_NAME_LENGTH) 's'],fn{fidx} ), 'char');
	fwrite(fid,fn{fidx},'char');
	if fldnameLenRes>0
		fwrite(fid,zeros(1,fldnameLenRes,'uint8'),'uint8');
	end
	
	className=class(fld);
	fwrite(fid,data_type(className),'uint8');
	fwrite(fid, fld, className);
end

fclose(fid);

end

function dtId=data_type(className)
switch(className)
	case 'double'
		dtId=0;
	case 'float'
		dtId=1;
		
	otherwise
		error('Not yet supported');
	
end

end