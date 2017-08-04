function [ gr ] = repairGraph( gr, bMoveDropoutToLph, dropoutDefaultType)
%REPAIRGRAPH Summary of this function goes here
%   This function must solve the following 2 tasks:
%1. check and repair where possible the NN graph connectivity
%2. find out correct PHL parameters for each LPH in the graph
%We're going to walk over the graph in a forward propagation manner starting from the first node and
%to recover necessary info and check connectivity
assert(isempty(dropoutDefaultType) || ischar(dropoutDefaultType));

LT=static_layer_types();
nodesCnt = length(gr);
assert(nodesCnt>1);
assert(nodesCnt == length(unique([gr(:).iid])));
assert(nodesCnt == length(unique({gr(:).name})));
assert(all(LT.unk ~= [gr(:).type] ));
assert(strcmp(gr(1).name,'Data'),'First item must be a Data source!');
assert(0==gr(1).parentIid);

gr = sortFields(gr);

gr(1).out_map=[];
%gr(1).lph_phl=[];
gr(1).phl_ref=[];
gr(1).visited=1;%to check if we've used the node
gr(1).innerIn=[];

gr=fprop_src(gr);
%looking for a second top level element
tli = find([gr(2:end).parentIid]==0);
assert(~isempty(tli));
assert(length(tli)==1,'Only one non-source data top level element permitted!');
tli = tli+1;
gr=run_fprop(gr,tli, bMoveDropoutToLph, false, dropoutDefaultType);

assert(all( arrayfun(@(v)~isempty(gr(v).visited), 1:nodesCnt ) ), 'There is a least one node in the graph, that was not visited during fprop walkarong. Wrong graph connections probably');
assert(all([gr( [gr(:).type]==LT.lfc ).nc] > 0));

gr = rmfield(gr, {'visited','x','y'});
gr = fill_phl(gr, tli, gr, 1:nodesCnt);
end

%%
function ri = fakeIids2real(gr, fakeGr, Iids)
grLen = length(gr);
ri = zeros(size(Iids));

for ii=1:length(Iids)
	if Iids(ii)>grLen
		ri(ii) = fakeGr(Iids(ii)).origIid;
	else
		ri(ii)=Iids(ii);
	end	
end

end

%%
function gr=fill_phl(gr, nodeId, fakeGr, fakeMap)
assert(length(fakeGr)==length(fakeMap));

LT=static_layer_types();

chList = gr(nodeId).childList;
chListLen = length(chList);

bThisNodeIsLPH = LT.lph==gr(nodeId).type;

if bThisNodeIsLPH
	%fill the .lph_phl and after that add 'fake' layers into fakeMap
	% все слои, указанные в .innerIn, должны строго совпадать с одним из .out_map какого-то
	% родительского слоя, находящегося строго под текущим слоем.
	% Находим по одному из слоёв из .innerIn этот самый родительский слой с соответстующим .out_map,
	% проверяем, что он строго под нами; проверяем, чтобы слои из .innerIn покрывали бы этот слой полностью;
	% затем заполняем на его основе .lph_phl.
	% Далее надо как-то сгруппировать нейроны, отдаваемые вложенным слоям, чтобы их можно было впоследствии
	% адресовать. Для этого смотрим .innerIn каждого дочернего слоя, проверяем, чтобы они образовывали бы
	% непрерывный диапазон и для каждого .innerIn создаём фейковый lph слой и фейковые исходные слои,
	% содержащие целиком требуемые .innerIn. Сохраняем соответствие этих ID скопированным так, чтобы
	% впоследствии обработка .innerIn использовала бы эти новые ID вместо реально указанных.
	assert(~isempty(chList) && ~isempty(gr(nodeId).innerIn));
	assert( all(gr(nodeId).innerIn == unique( [gr(chList).innerIn] )) );
	fakeSrcIds = fakeMap( gr(nodeId).innerIn );
	sortedFakeSrcIds = sort(fakeSrcIds);
	innerInCnt = length(fakeSrcIds);
	
	if innerInCnt==1 && ~isLayerAtLPHTop(fakeGr, fakeSrcIds)
		%using layer itself
		srcParentId = fakeSrcIds;
	else
		%source layer must be on top of PHL
		assert(innerInCnt>1 || isLayerAtLPHTop(fakeGr, fakeSrcIds), ['Source layer of ' gr(nodeId).name ' must lie on the top of LPH']);
		%looking for a parent node, whose .out_map equals to fakeSrcIds
		srcParentId = fakeGr(fakeSrcIds(1)).parentIid;
		if innerInCnt>1
			while srcParentId>0
				if innerInCnt==length(fakeGr(srcParentId).out_map) && all(sortedFakeSrcIds == sort(fakeGr(srcParentId).out_map))
					break;
				else
					srcParentId = fakeGr(srcParentId).parentIid;
				end			
			end
			assert(srcParentId>0,['Failed to find common parent node for .innerIn of ' gr(nodeId).name]);
		end
	end
	oMap = fakeGr(srcParentId).out_map;
	oMapLen = length(oMap);
	
	assert(innerInCnt==oMapLen && all(sortedFakeSrcIds == sort(oMap)), ['.innerIn of ' gr(nodeId).name ' points to a wrong layer']);
	%% todo: we should check here if the nodeId lies directly on the top of srcParentId
	% (but this check looks fairly complicated in a general case)
	
	%making .lph_phl
	fakeGrLen = length(fakeGr);
	oMapIdxs = zeros(1,fakeGrLen);
	%oMapOffs = zeros(oMapLen,1);
	oMapOffRefs = cell(oMapLen,1);
	%nc=0;
	for ii=1:oMapLen
		%oMapOffs(ii) = nc;
		if ii>1
			oMapOffRefs{ii} = [oMapOffRefs{ii-1} oMap(ii-1)];
		else
			oMapOffRefs{ii} = 0;
		end
		%nc = nc + fakeGr(oMap(ii)).nc;
		oMapIdxs( oMap(ii) )=ii;
	end
	
	%gr(nodeId).lph_phl = zeros(chListLen,2);
	gr(nodeId).phl_ref = cell(chListLen,2);
end

if ~isempty(chList)
	for ii=1:chListLen
		if bThisNodeIsLPH
			%fixing fakeMap and fakeGr to reflect necessary sub-layers
			chIn = fakeMap( gr(chList(ii)).innerIn );
			chInLen = length(chIn);
			
			assert(chInLen==length(intersect(chIn, oMap)) );
			% chIn is an (unordered) set of ids of layers listed as incoming into i-th child of nodeId.
			% All of chIn layers are listed in the oMap list.
			% We had to check whether chIn layers form a continuous subset of oMap (i.e. these
			% layers must be listed in the (ordered) oMap together)
			if chInLen>1
				chInOMapIdxs = sort(oMapIdxs(chIn));
				assert( all(chInOMapIdxs > 0) );
				assert( all( 1==diff(chInOMapIdxs) ), ['Child ' gr(chList(ii)).name ' of the layer ' gr(nodeId).name ...
					' has discontinuous list of incoming nodes'] );
			else
				chInOMapIdxs = oMapIdxs(chIn);
			end
			oMapSpan = chInOMapIdxs(1) : chInOMapIdxs(end);
			%gr(nodeId).lph_phl(ii,:) = [oMapOffs(chInOMapIdxs(1)) sum([fakeGr( oMap(oMapSpan) ).nc]) ];	
			gr(nodeId).phl_ref{ii,1} = fakeIids2real(gr, fakeGr, oMapOffRefs{chInOMapIdxs(1)});
			gr(nodeId).phl_ref{ii,2} = fakeIids2real(gr, fakeGr, oMap(oMapSpan));
			
			if chInLen < oMapLen
				% we have to add "fake" layers into fakeGr and fakeMap to enable inner layers processing
				%adding the parent layer
				newFakeGr = fakeGr;
				
				newFakeGrLen = fakeGrLen+1;
				newFakeMap = [fakeMap (newFakeGrLen:newFakeGrLen+chInLen) ];
				fakeParentId = newFakeGrLen;
				newFakeGr(fakeParentId) = newFakeGr(srcParentId);
				newFakeGr(fakeParentId).iid = fakeParentId;
				newFakeGr(fakeParentId).parentIid = 0;%this shouldn't point to a normal layer to catch errors early
				%adding necessary .out_map layers
				for jj=1:chInLen
					lIidToFake = newFakeGr(oMap(oMapSpan(jj))).iid;
					newFakeGrLen = newFakeGrLen+1;
					if lIidToFake > length(gr)
						newFakeMap( lIidToFake ) = newFakeGrLen;
						lIidToFake = newFakeGr(lIidToFake).origIid;
						assert(~isempty(lIidToFake) && lIidToFake < length(gr));
					end
					newFakeMap( lIidToFake ) = newFakeGrLen;
					newFakeGr(newFakeGrLen) = newFakeGr(lIidToFake);
					newFakeGr(newFakeGrLen).iid = newFakeGrLen;
					newFakeGr(newFakeGrLen).parentIid = fakeParentId;				
					newFakeGr(newFakeGrLen).origIid = lIidToFake;
				end

				%fixing the .out_map
				newFakeGr(fakeParentId).out_map = newFakeMap(oMap(oMapSpan));
				
				gr = fill_phl(gr, chList(ii), newFakeGr, newFakeMap);
			else
				gr = fill_phl(gr, chList(ii), fakeGr, fakeMap);
			end
		else
			gr = fill_phl(gr, chList(ii), fakeGr, fakeMap);
		end
	end
end

end

%%
function gr=run_fprop(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType)
LT=static_layer_types();

assert(isempty(gr(nodeId).visited), 'Too bad, tried to fprop the layer twice!');
gr(nodeId).visited=1;

switch(gr(nodeId).type)
	case LT.src
		error('Mustnt find a source node here!');
		
	case LT.lpv
		gr=fprop_lpv(gr, nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType);
		
	case LT.lph
		gr=fprop_lph(gr, nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType);
		
	otherwise
		gr=fprop_simple(gr, nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType);
end
end
%%
function gr=fprop_simple(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType)
% it's rather generic function that can be called for compound layers also
LT=static_layer_types();
parentIid = gr(nodeId).parentIid;

%% incoming links
inList = gr(nodeId).in;
inListLen = length(inList);
for ii=1:inListLen
	assert( ~isempty(gr(inList(ii)).visited), ['The link into `' gr(nodeId).name '` goes from not yet visited `' gr(inList(ii)).name '`']);	
end

if isLayerTypeCompound(gr(nodeId).type)
	assert( isempty(gr(nodeId).nc), 'Compound layer cant have .nc property set here');
	assert(length(gr(nodeId).childList) > 1);
	assert(inListLen<=1 || (inListLen>1 && isLayerAtLPHBottom(gr,nodeId)),...
	['Compound node may have zero or one in link. It may have more than 1 incoming link only if it lives at the bottom of a LPH. Check ' gr(nodeId).name]);
else
	assert(parentIid > 0);
	
	if bMoveDropoutToLph && bUpperLphHasDropout
		%dropping
		gr(nodeId).dropout=[];
	end
	
	switch(gr(nodeId).type)
		case LT.lid
			assert(isempty(gr(nodeId).nc), 'Identity layer mustn`t have the .nc property set!');
			if ~isempty(gr(nodeId).dropout) && ~(islogical(gr(nodeId).dropout) && ~gr(nodeId).dropout)
				if isnumeric(gr(nodeId).dropout) && 1==gr(nodeId).dropout
					%it was turned on by default. Turning it off
					gr(nodeId).dropout=[];
				else
					disp(['* Warning: LID "' gr(nodeId).name '" has dropout property=' gr(nodeId).dropout '. You could end up applying dropout twice!']);
				end
			end
			
			%{
			ncArr = [gr(inList).nc];
			assert( all(ncArr>0), ['Every link incoming into the LID must have the .nc property set. See ' gr(nodeId).name] );
			nc = sum(ncArr);
			if gr(nodeId).nc>0
				assert(nc == gr(nodeId).nc, ['Error: LID node ' gr(nodeId).name ' have different .nc value property set and calculated from incoming links']);
			else
				gr(nodeId).nc=nc;
			end
			%}
		case LT.custom
			if isempty( gr(nodeId).constr )
				assert( ~isempty(gr(nodeId).nc) && ((isnumeric(gr(nodeId).nc) && gr(nodeId).nc>0) || ischar(gr(nodeId).nc)), ...
					['Non compound layer must have the nc property set. Check ' gr(nodeId).name]);
			else
				assert(isempty(gr(nodeId).nc), 'Custom layer with .constr property mustn`t have .nc property set!');
				if ~isempty(gr(nodeId).dropout)
					%it was turned on by default. Turning it off if there's a .constr property set
					if ~isempty(gr(nodeId).constr)
						assert(isnumeric(gr(nodeId).dropout) && 1==gr(nodeId).dropout, 'Looks like the dropout was turned on by hand for a .constr layer. It does not affect on the layer!');
						gr(nodeId).dropout=[];
					end
				end
			end
			
		otherwise
			assert( ~isempty(gr(nodeId).nc) && ((isnumeric(gr(nodeId).nc) && gr(nodeId).nc>0) || ischar(gr(nodeId).nc)), ...
				['Non compound layer must have the nc property set. Check ' gr(nodeId).name]);
	end
	gr(nodeId).out_map = nodeId;
	
	assert(isempty(gr(nodeId).childList));
	
	assert(inListLen==1 || (inListLen>1 && isLayerAtLPHBottom(gr,nodeId)),...
	['Simple node could have only one incoming link. It could have more than 1 incoming link only if it lives at the bottom of a LPH. Check ' gr(nodeId).name]);
end

if ~isempty(gr(nodeId).dropout) && isnumeric(gr(nodeId).dropout) && 1==gr(nodeId).dropout
	gr(nodeId).dropout = dropoutDefaultType;
end

%% TODO if there are more than 1 incoming link, all of them must reside in a single LPH directly under the current
% node - but it's fairly hard to check in a most generic case


gr(nodeId).innerIn = inList;

%% outgoing links
outList = gr(nodeId).out;
outListLen = length(outList);
for ii=1:outListLen
	assert( isempty(gr(outList(ii)).visited), 'Outgoing node mustnt be visited' );	
end
%if the layer lies on the top of a LPV, we should move our out links to that LPV
if parentIid>0 && outListLen>0 && LT.lpv == gr(parentIid).type && nodeId == gr(parentIid).childList(end)
	%update .in destination to parentIid
	for ii=1:outListLen
		idxs=find( gr(outList(ii)).in==nodeId,1 );
		assert(~isempty(idxs));
		gr(outList(ii)).in(idxs) = parentIid;
		%% todo probably we should also sort .in, but do we really need it sorted?
	end
	
	%change .out
	newOut = unique([gr(parentIid).out outList]);
	gr(nodeId).out=[];
	outList=[];
	outListLen=0;	
	[~,idxs]=sort([ gr(newOut).x ]);
	gr(parentIid).out = newOut(idxs);
end
switch(outListLen)
	case 0
		%we must be on the top of a compound layer here or inside of LPV
		assert(parentIid==0 || LT.lpv==gr(parentIid).type || isLayerOnTopOfCompound(gr, nodeId), ...
			['All layers must have outgoing links except those that resides on a top of compound layers. See ' gr(nodeId).name]);
		
	case 1
		% we must be inside LPV below the top of the stack here, or on the top of LPH
		if ~isLayerAtLPHTop(gr,nodeId)
			assert(parentIid>0 && LT.lpv == gr(parentIid).type && nodeId ~= gr(parentIid).childList(end)...
				&& parentIid==gr(outList(1)).parentIid,...
				['Invalid outlink in ' gr(nodeId).name]);
		end
		
	otherwise
		%we must be directly under a LPH
		assert(isLayerUnderLPH(gr,nodeId),['Only a layer under a PHL is allowed to have multiple outgoing links. See ' gr(nodeId).name]);
		
end

end

%%
function b=isLayerUnderLPH(gr,nodeId)
LT=static_layer_types();
b=false;
pId = gr(nodeId).parentIid;
while pId>0
	switch(gr(pId).type)
		case LT.lph
			nodeId = pId;
			pId = gr(nodeId).parentIid;
			
		case LT.lpv
			chI = find( gr(pId).childList == nodeId, 1 );
			assert(~isempty(chI));			
			if chI == length(gr(pId).childList)
				nodeId = pId;
				pId = gr(nodeId).parentIid;
			else
				b = ( gr( gr(pId).childList(chI+1) ).type == LT.lph );
				return;
			end
			
		otherwise
			error('Unexpected parent layer type');	
	end
end
end

function b=isLayerOnTopOfCompound(gr,nodeId)
LT=static_layer_types();
b=false;
pId = gr(nodeId).parentIid;
if pId>0
	switch(gr(pId).type)
		case LT.lph
			b=true;			
			
		case LT.lpv
			if nodeId == gr(pId).childList(end)
				b=true;
			end
			
		otherwise
			error('Unexpected parent layer type');	
	end
end
end

function b=isLayerAtLPHTop(gr,nodeId)
LT=static_layer_types();
b=false;
pId = gr(nodeId).parentIid;
while pId>0
	switch(gr(pId).type)
		case LT.lph
			b=true;
			return;			
			
		case LT.lpv
			if nodeId == gr(pId).childList(end)
				nodeId = pId;
				pId = gr(nodeId).parentIid;
			else
				return;
			end
			
		otherwise
			error('Unexpected parent layer type');	
	end
end
end

function b=isLayerAtLPHBottom(gr,nodeId)
LT=static_layer_types();
b=false;
pId = gr(nodeId).parentIid;
while pId>0
	switch(gr(pId).type)
		case LT.lph
			b=true;
			return;			
			
		case LT.lpv
			if nodeId == gr(pId).childList(1)
				nodeId = pId;
				pId = gr(nodeId).parentIid;
			else
				return;
			end
			
		otherwise
			error('Unexpected parent layer type');	
	end
end
end

%%
function gr=fprop_lph(gr, nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType)
gr=fprop_compound(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout || ~isempty(gr(nodeId).dropout), dropoutDefaultType);
%0. проверяем/исправляем собственные связи
%1. проходим по всем детям и запускаем fprop у них - там просто проверяем/исправляем связность и заполняем
%свойства .innerIn, и .visited
%2. когда дети проверены/настроены, обновляем собственные свойства: .nc, .innerIn, .out_map и .lph_phl
%gr(nodeId).nc = sum([gr( gr(nodeId).childList ).nc]);
gr(nodeId).innerIn = unique( [gr(nodeId).innerIn gr( gr(nodeId).childList ).innerIn] );

gr(nodeId).out_map=[ gr( gr(nodeId).childList ).out_map ];

end
%%
function gr=fprop_lpv(gr, nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType)
assert(isempty(gr(nodeId).dropout) && isempty(gr(nodeId).lossAdds),'WTH would you need dropout or LA here?');
gr=fprop_compound(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType);

topNodeId = gr(nodeId).childList(end);
%gr(nodeId).nc = gr( topNodeId ).nc;

if isLayerTypeCompound(gr(topNodeId).type)
	gr(nodeId).out_map= gr( topNodeId ).out_map;
else
	gr(nodeId).out_map = nodeId;
end
gr(nodeId).innerIn = unique([gr(nodeId).innerIn gr( gr(nodeId).childList(1) ).innerIn]);

end

function gr=fprop_compound(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType)
assert( isempty(gr(nodeId).nc),'Compound layer cant have the nc property set here!');
%check/update links
gr=fprop_simple(gr,nodeId, bMoveDropoutToLph, bUpperLphHasDropout, dropoutDefaultType);
%walk over the child list
LT=static_layer_types();
bLpv = gr(nodeId).type==LT.lpv;
chList = gr(nodeId).childList;
lchl = length(chList);
for ii=1:lchl
	gr=run_fprop(gr, chList(ii), bMoveDropoutToLph, (~bLpv || ii==lchl) && bUpperLphHasDropout , dropoutDefaultType);	
end
end

%%
function gr=fprop_src(gr)
LT=static_layer_types();
gr(1).dropout=[];
if gr(1).type==LT.src
	assert( ~isempty(gr(1).nc) && ((isnumeric(gr(1).nc) && gr(1).nc>0) || ischar(gr(1).nc) ));
	gr(1).out_map=1;
	assert(isempty(gr(1).in));
	assert(~isempty(gr(1).out));
else
	assert(gr(1).type==LT.lph);
	chList = gr(1).childList;
	assert( all( [gr(chList).type] == LT.src ), 'All source Data package children must have LT.src type' );
	for ii=1:length(chList)
		gr(chList(ii)).visited=1;
		gr(chList(ii)).dropout=[];
		assert( ~isempty(gr(chList(ii)).nc) && ((isnumeric(gr(chList(ii)).nc) && gr(chList(ii)).nc>0) || ischar(gr(chList(ii)).nc) ));
	end
	
	%gr(1).nc = sum([gr(chList).nc]);
	gr(1).out_map = chList;
	
	assert( all( arrayfun(@(ce)isempty( gr(ce).in ), chList) ),'There must be no incoming connections into source data' );
	assert( all( arrayfun(@(ce)~isempty( gr(ce).out ), chList) ),'There must be an outgoing connections from source data elements' );
end
end




%%
function gr = sortFields(gr)
%sorts childList according to children position and a corresponding parent layer type; sorts in/out

LT=static_layer_types();
for ii=1:length(gr)
	isComp = isLayerTypeCompound(gr(ii).type);
	childList = gr(ii).childList;
	assert( xor(isComp, isempty(childList)), ['Incorrect layer type for the childList for ' gr(ii).name]);
	assert( ~isComp || (length(childList)>1 && length(childList)==length(unique(childList))),...
		['Compond layers must have more than 1 children without repetitions. See ' gr(ii).name] );
	if ~isempty(childList)
		switch(gr(ii).type)
			case LT.lpv
				sortMe = -[gr(childList).y];
				
			case LT.lph
				sortMe = [gr(childList).x];
				
			otherwise
				error(['Non-empty childList for non compound layer. See ' gr(ii).name]);
		end
		[~,idxs]=sort(sortMe);
		gr(ii).childList = childList(idxs);
	end
	if length(gr(ii).in)>1
		[~,idxs]=sort([ gr(gr(ii).in).x ]);
		gr(ii).in = gr(ii).in(idxs);
	end
	if length(gr(ii).out)>1
		[~,idxs]=sort([ gr(gr(ii).out).x ]);
		gr(ii).out = gr(ii).out(idxs);
	end
end

%gr = rmfield(gr,'y');%we'll need .x to make correct connection updates in future
end





%{
ПРАВИЛА:
- входящие связи могут приходить только в простые ноды. Если связь идёт в составную ноду, то иногда
(для LPV, например) её можно правильно перенаправить на соответвующую внутреннюю простую ноду.
Если это сделать нельзя (для LPH, например) - ошибка.
- в каждую простую ноду может входить больше одной входящей связи только если эта простая нода в
конечном итоге является низом LPH и все её входящие связи формируют непрерывный диапазон нейронов.
- для составных нод следует отдельно хранить список входящих в их внутренние ноды
связей. Исходящие ноды этого списка должны образовывать непрерывный диапазон нейронов (требование
LPH), но не обязаны быть сортированы каким либо образом (LPH позволяет произвольно адресовать
непрерывные поддиапазоны нейронов)

- исходящие связи могут выходить как из простых, так и из составных нод. При этом:
--- непосредственно внутри LPV исходящие связи могут быть строго от одного узла к следующему без
перескоков. За пределы LPV (т.е. от самого верхнего узла)
связей быть не может, однако если LPV является верхней частью LPH и верхняя нода LPV содержит
исходящие связи, эти связи должны быть "переписаны" на самый верхний LPV этой ноды внутри данного
LPH. В остальных случаях - ошибка. 
--- непосредственно внутри LPH исходящих связей во внутрь этого LPH быть не может, однако могут быть
исходящие связи во-вне.
--- любая нода может иметь несколько исходящих связей только если каждая из этих связей входит
внутрь одного и того же единственного LPH, содержащего в самом низу целевые ноды связи.
--- из любой ноды (не считая нетоповой простой ноды внутри LPV) исходящая связь может быть только
когда эта нода является верхней нодой в LPH (т.е. из неё нет внутренних исходящих связей) и идёт
внутрь _другого_ LPH (только он может быть сможет её правильно адресовать). В остальных случаях
исходящие связи невозможны. 

АЛГО:
- проходим граф связей от каждой исходной ноды вглубь до нахождения конца.
- в процессе хождения запоминаем каждую затронутую ноду и проверяем, чтобы в любом простом слое мы бы
побывали строго не больше одного раза. 
- заходя в каждую последующую ноду:
--- у составной ноды (кроме Data) должно быть минимум две дочерних ноды. В списке дочерних элементов
повторов быть не должно.
--- у простой ноды обязан быть указан nc. Простая нода (кроме src) должна быть вложена в составную
(ибо иначе весь этот огород нах не нужен).
--- для простой целевой ноды поднимаемся по стеку родительских нод, добавляя в их набор внутренних
входящих связей источник и обновляя их кол-во выходных нейронов.
--- для составной целевой ноды ищем реальный целевой несоставной слой и перенаправляем связь ему
(т.е. удаляем связь у себя и создаём ему), затем от этого узла делаем предыдущий пункт.
--- проверяем выполнение правил связности для данной ноды
--- проходим по следующей исходящей связи из текущей ноды. Когда исходящей связи нет, поднимаемся на
шаг вверх к родителю и продолжаем движение из него.
- агрегируем посещения всех узлов по каждому проходу из каждой исходной ноды и в конце проверяем,
чтобы каждая несоставная нода была бы пройдена хоть один раз.
- после завершения обхода делаем просто проход по всем нодам, где строго проверяем выполнение правил
связности для каждой ноды графа (т.к. мы могли не проверить их для всех составных нод, либо в
процессе обхода могли что-то изменить и не проверить), и вычисляем необходимые смещения для LPH/PHL
(это нельзя делать в первый обход, поскольку необходимо гарантировать корректность значений nc и
innerIn для каждой ноды, а они в первый обход ещё только формируются).
%}
