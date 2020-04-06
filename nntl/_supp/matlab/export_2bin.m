function export_2bin( S, fname, seqDescr, bDropUnknown)
%EXPORT_STRUCT_2BIN Export 2D matrix or struct to NNTL binary file
% (see nntl/_supp/io/binfile.h for specifications)

MAX_FIELD_NAME_LENGTH=15;

bSeqMode = exist('seqDescr','var') && ~isempty(seqDescr);
if bSeqMode
	assert(isvector(seqDescr));
	bDropUnknown=false;
else
	seqDescr=[];
	bDropUnknown = ~exist('bDropUnknown','var') || logical(bDropUnknown);
end
nClassesCnt = numel(seqDescr);

if ~isstruct(S) && ismatrix(S)
	S=struct('mtx',S);
	bDropUnknown = false;
end
if ~isstruct(S)
	error('S must be a 2D matrix or struct');
end

%%
[fid,err]=fopen(fname,'w','l');
if -1==fid
	error('Failed to open file %s: %s\n',fname,err);
end

fn = fieldnames(S);
fc = length(fn);

%bin_file::HEADER
fwrite(fid,'nntl','char');

% format version number
fwrite(fid, 0,'uint16');

%WORD wFieldsCount;//total count of all fields besides HEADER
fwrite(fid,fc + bSeqMode*nClassesCnt,'uint16');

%WORD wSeqClassCount;//if non zero, then expecting file to contain a set of sequences belonging to this amount of classes
%//there must be this count of CLASS_ENTRY structures immediately after HEADER
if bSeqMode
	assert(nClassesCnt>0 && nClassesCnt< 2^16);
	fwrite(fid, nClassesCnt,'uint16');
	
	%writing bin_file::CLASS_ENTRY
	fwrite(fid, seqDescr, 'uint16');
else
	fwrite(fid, 0, 'uint16');
end

%% bin_file::FIELD_ENTRY
for fidx=1:fc
	fld = S.(fn{fidx});
	
	if bDropUnknown
		switch fn{fidx}
			case {'train_x','train_y','test_x','test_y'}				
			otherwise
				continue;
		end
	end
	
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

fprintf(1,'Done with %s !\n', fname);

end

function dtId=data_type(className)
switch(className)
	case 'double'
		dtId=0;
	case 'single'
		dtId=1;
		
	otherwise
		error('Not yet supported');
	
end

end