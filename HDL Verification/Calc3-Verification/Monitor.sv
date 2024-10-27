`ifndef GUARD_MONITOR
`define GUARD_MONITOR
`include "Interface.sv"
class Monitor; 
virtual GlobalPorts globalPorts_;
virtual Output out_ports[1:4];
function new (virtual GlobalPorts globalPorts_, virtual Output port1, virtual Output port2, virtual Output port3,virtual Output port4);
begin
    this.globalPorts_=globalPorts_;
    this.out_ports[1]=port1;
    this.out_ports[2]=port2;
    this.out_ports[3]=port3;
    this.out_ports[4]=port4;
end
endfunction
task start();
begin
   $display("Monitor started");
    fork
        scan(1);
        scan(2);
        scan(3);
        scan(4);
    join
end
endtask
task scan (int _pNumber);
forever fork
    @(posedge this.globalPorts_.clock or negedge this.globalPorts_.clock);
    if (this.out_ports[_pNumber].outx_resp !=0) begin
        if (this.globalPorts_.clock==0) begin
            $display ("Monitor_Error : RespOut on negedge,");
        end
    end
    if (this.out_ports[_pNumber].outx_resp !=1) begin
        if (this.out_ports[_pNumber].outx_data!=32'bz)begin
            $display ("Monitor_Error : wrong data fetch,");
        end
    end
    

join
endtask
endclass
`endif