% Synfire chain analysis
% Aaron Miller - 5/31/2008
ex = 99;
while ex~=0

fprintf('\n\n\n\n*****Generate plots from data files created by synfireGrowth.cpp.*****\n');
fprintf('1. Draw network topology\n');
fprintf('2. Display distribution of synaptic weights\n');
fprintf('3. Generate roster plot\n');
fprintf('4. Plot synaptic weights over range of trials\n');
fprintf('0. Quit\n');
option = input('\nPlease select one of the above options or to quit:\n');

switch option
    
    case 1 %draw network topology
        clear;
        ex = 1;
        trial = input('\nWhich trial?\n(To display sequence of networks, enter trial range in MATLAB array notation, \n----- i.e., begin:interval:end)\n');
        runid = input('\nWhich run?\n');
        
        for t=1:length(trial)
        raw = load(sprintf('connect/connect%.0fr%.0f.dat',trial(t),runid));
        if isempty(raw)
            figure;
            axis([0 1 0 1]);
            text(.4,.5,sprintf('Trial %.0f: No network to display!!',trial(t)));
            ques = 1;
            upperg = -4;
            continue;
        end
        
        % i indexes group number, grsz(i) is the size of ith group
        % g is a structure storing all connectivity info
        % g(i).mem stores all members of ith group (pre)
        % g(i).issat(j) stores a 1 if g(i).mem(j) is saturated
        % g(i).cns{j} stores the postsynaptic cns of g(i).mem(j)
        % g(i).grcns{j} stores the group rankings of members of g(i).cns{j}
        % ALL NEURON NUMBERS IN G(i) CORRESPOND TO LABELS IN C++ PROG

        %determine connectivities from raw
        %format of input: <pre> <pre_group> <post> <post_group> <post sat?>
        %...<G[pre][post]
        max_rank = max(raw(:,4));
        %all members of group 1 are pre
        g(1).mem = union(raw(find(raw(:,2)==1),1),raw(find(raw(:,2)==1),1));
        grsz(1) = length(g(1).mem);
        g(1).issat = ones(grsz(1),1);
        %loop thru all other groups, who must be post
        for i=1:max_rank
            
            if (i~=1)
            g(i).mem = union(raw(find(raw(:,4)==i),3),raw(find(raw(:,4)==i),3)); %labels of neurons group i  
            grsz(i) = length(g(i).mem); %# of neurons in group i
            g(i).issat = raw(find(raw(:,4)==i),5); 
            end
            
            for j=1:grsz(i)
        
                g(i).cns{j} = raw(find(raw(:,1)==g(i).mem(j)),3); %labels of neurons that g(i).mem(j) connects to
                g(i).grcns{j} = raw(find(raw(:,1)==g(i).mem(j)),4); %group ranks for neurons that g(i).mem(j) connects to
        
            end
            
        end

        clear raw;
        
        fprintf('Load successful.\n');
        %initial values for MATLAB
        if t==1
            ques=0;
            upperg=1;
        end
        
        for rep = 1:20 %ask to redraw network 20 times unless loop is broken, breaks automatically if length(trial)~=1
        
        %setup length of chain displayed 
        if t==1 || (ques==1 && max_rank~=upperg) || (ques==0 && max_rank<upperg)
            
            lowerg = max_rank;
            upperg = 1;
            if t~=1
                beep
                fprintf('\n\nPlease select new display options.');
            end
            fprintf('\n# of groups in chain is %.0f.\n',max_rank);
            while (lowerg>=upperg)
                lowerg = input('\nSmallest group number to display?\n'); %Possibility to display a portion of the chain
                upperg = input('\nLargest group number to display?\n');
                if upperg>max_rank || lowerg>=upperg
                    fprintf('Display range invalid.\n');
                end
            end
        
            if length(trial)~=1
                ques = input('\nInclude more groups in later pictures? (No = 0, Yes = 1, Always include all groups = 2)\n');
            end
            
        end
        
        if ques == 2
            upperg=max_rank;
        end

        %handle figure
        if t==1
            figure;
        else;
            clf;
        end
        
        Gs2disp = lowerg:upperg;
        
        %determine plot size and assign coordinates
        xspace = 1/(max(grsz(Gs2disp))+1); %x spacing between neurons
        yspace = 10*xspace; %y spacing is 10x the x spacing so that the pic is pleasing to the eyes
        axis([0 1 0 yspace*(length(Gs2disp)+1)]);

        c=1;
        for i = fliplr(Gs2disp)
    
            g(i).y = c*yspace;
            xst = .5-.5*xspace*(length(g(i).mem)-1);
            
            for j = 1:length(g(i).mem)
        
                g(i).x(j) = xst + (j-1)*xspace;
        
            end
            c=c+1;
        end

        n_incorp = 0;
        %draw lines
        for i=Gs2disp
            n_incorp = n_incorp + grsz(i);
            for j=1:length(g(i).cns)
        
                for k = 1:length(g(i).cns{j})
            
                    cnto = g(i).grcns{j}(k);
                    switch 1 %set line color depending on the group of the post
                        case cnto>i+1
                            lcol = 'r';
                        case cnto<=i
                            lcol = 'b';
                        otherwise
                            lcol ='g';
                    end
            
                    switch 1 %in case post is not in display range
                        case cnto>upperg
                            ytarget = -5;
                            xtarget = .5;
                    
                        case cnto<lowerg
                            ytarget = length(grsz)*yspace;
                            xtarget = .5;
                    
                        otherwise
                            xtarget = g(cnto).x(find(g(cnto).mem==g(i).cns{j}(k)));
                            ytarget = g(cnto).y;
                    end
            
                    line([g(i).x(j) xtarget],[g(i).y ytarget],'Color',lcol);
            
                end
            end
        end

        %draw markers
        hold on;
        for i = Gs2disp
    
            for j=1:length(g(i).mem)
        
                if g(i).issat(j)
                    dotc='k';
                else
                    dotc='c';
                end
                plot(g(i).x(j),g(i).y,'o','MarkerFaceColor',dotc);
                text(g(i).x(j),g(i).y-xspace,int2str(g(i).mem(j)),'HorizontalAlignment','center','VerticalAlignment','top');
            end
        end

        title(sprintf('Trial %.0fr%.0f, %.0f neurons in Groups %.0f-%.0f',trial(t),runid,n_incorp,lowerg,upperg));
        hold off;   
        
        if length(trial)==1
            again = input('\nMake another plot with this trial? (no = 0, yes = 1)\n');
        
            if again == 0 break;
                
            else
              for i=1:max_rank
                clear g(i).x;
              end
            end
        else
            break;
        end
        
        end
        clear Gs2disp;
        clear g;

        pause;
        end
        
    case 2 %distribution of weights
        clear;
        ex = 2;
        trial = input('\nWhich trial?\n');
        runid = input('\nWhich run?\n');
        cap = input('\nWhat was the max synaptic strength of this run (the default set in synfire.h is .6)?\n');
        bin_size = input('\nChoose bin size (smaller bin means better resolution but longer plot time, recommended size is .01):\n');
        G = load(sprintf('syn/syn%.0fr%.0f.dat',trial,runid));
        
        bins = [0:bin_size:cap];
        figure;
        
        old = [];
        for i=2:length(bins)
    
            new=find(G<=bins(i));
            n_in_bin(i-1)=length(setxor(new,old));
            clear old;
            old = new;
    
        end
        
        clear G;
        
        axis([0 cap+.1 0 max(n_in_bin)+200]);

        for i=1:length(n_in_bin)
            
            line([bins(i) bins(i)],[0 n_in_bin(i)]);
            line([bins(i+1) bins(i+1)],[0 n_in_bin(i)]);
            line([bins(i) bins(i+1)],[n_in_bin(i),n_in_bin(i)]);
            
        end
        
        title(sprintf('Distribution of synaptic weights, trial %.0fr%.0f',trial,runid));
        
    case 3 %roster plot
        clear;
        ex=3;
        trial = input('\nWhich trial?\n');
        runid = input('\nWhich run?\n');
        raw = load(sprintf('roster/roster%.0fr%.0f.dat',trial,runid));
        ts = input('Time range lower bound (in ms): ');
        te = input('to time range upper bound (in ms): ');
        fprintf('\nDisplay option:\n1. Plot spikes by neuron label\n2. Plot spikes by group #\n ');
        opt = input('?\n');
        %format of input: <label> <group> <spike time>
        ymax = max(raw(:,opt))+2;

        figure;
        axis([ts te 0 ymax]);
        spk_c=0;
        for i=1:size(raw,1);
            if raw(i,3)>=ts && raw(i,3)<=te
                spk_c=spk_c +1;
                if raw(i,2) == 0
                    c = 'r'; 
                else
                    c = 'b';
                end
                line([raw(i,3) raw(i,3)], [raw(i,opt)+.4 raw(i,opt)-.4], 'Color', c);
            end
        end
        
        title(sprintf('Roster Plot, trial %.0fr%.0f, %.0f spikes in this interval out of %.0f total this trial', trial,runid,spk_c,size(raw,1)));
        clear raw;
    
    case 4 %time dep of synapses
        clear;
        ex=4;
        trial = input('\nEnter trial range (use MATLAB array notation, i.e., begin:interval:end,\n with interval=(muliple of synapse_save) from synfire.cpp):\n');
        runid = input('\nWhich run?\n');
        fprintf('\nSpecify synapses to track by entering pairs of neuron labels [pre post].\nIndicate the end of the list by entering 0.\n');
        syn = [0 0];
        pre = [];
        post = []
        for i=1:100 %track up to 100 synapses
            fprintf('\nSynapses to track:\n');
            disp(pre);
            disp(post);
            syn = input('Enter new synapse: ');
            if length(syn)==2
                pre(i)=syn(1);
                post(i)=syn(2);
            else
                fprintf('\nSynapses to track (final):\n');
                disp(pre);
                disp(post);
                break;
            end
        end
        
        fprintf('Warning: Plot may take several minutes to draw, depending on the size of the trial range \nand the number of neurons in the network. ');
        fprintf('If MATLAB takes too long (or crashes), \ntry reducing the number of trials you sample by increasing the interval.');
        yvals = zeros(length(pre),length(trial));
        for i = 1:length(trial)
            G = load(sprintf('syn/syn%.0fr%.0f.dat',trial(i),runid));
            for j=1:length(pre)
                yvals(j,i)=G(pre(j)+1,post(j)+1);%'+1's due to the array labeling difference between c++ and MATLAB
            end
            clear G;
        end

        plot(trial,yvals);
        
        %Make legend
        for i=1:length(pre)
            leg(i,:) = sprintf('G[%3.0f][%3.0f]',pre(i),post(i));
        end
        
        legend(leg);
        title(sprintf('Synaptic weight vs. Trial, run %.0f',runid));
        
    case 0
        ex=0;
        
    otherwise
        fprintf('Selection is not a menu option.\n');
end

end
