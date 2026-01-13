"""
GWL Serializer - Converts Endpoint JSON to Nanoscribe GWL Format

Generates layer-by-layer .gwl files for Nanoscribe GWL writing.
Each layer becomes a separate GWL file with proper headers and formatting.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_gwl_header(gwl_params: Dict) -> str:
    """Generate standard GWL header with print parameters"""
    header = f"""GalvoScanMode
PowerScaling {gwl_params.get('power_scaling', 1.0)}
LaserPower {gwl_params.get('laser_power', 40)}
ScanSpeed {gwl_params.get('scan_speed', 100)}

FindInterfaceAt {gwl_params.get('find_interface_at', 0)}
XOffset {gwl_params.get('x_offset', 0)}
YOffset {gwl_params.get('y_offset', 0)}
ZOffset {gwl_params.get('z_offset', 0)}

"""
    return header


def serialize_layer_to_gwl(layer: Dict, gwl_params: Dict) -> str:
    """
    Convert a single layer to GWL format
    
    Args:
        layer: {z_um: float, segments: [{start: [x,y], end: [x,y]}, ...]}
        gwl_params: GWL printing parameters
        
    Returns:
        GWL string for this layer
    """
    gwl_lines = []
    
    # Add header
    gwl_lines.append(generate_gwl_header(gwl_params))
    
    # Add wait command
    wait_time = gwl_params.get('wait_time', 0.1)
    gwl_lines.append(f"Wait {wait_time}")
    
    # Get z coordinate
    z = layer['z_um']
    
    # Sort segments by Y-coordinate (bottom to top, left to right)
    # This ensures proper printing order: bottom of all circles first, then next layer
    sorted_segments = sorted(layer['segments'], key=lambda seg: seg['start'][1])
    
    # Process each segment individually with its own Write command
    for seg in sorted_segments:
        x_start, y_start = seg['start']
        x_end, y_end = seg['end']
        
        # Write start point
        gwl_lines.append(f"{x_start:.2f}\t{y_start:.2f}\t{z:.3f}")
        
        # Write end point
        gwl_lines.append(f"{x_end:.2f}\t{y_end:.2f}\t{z:.3f}")
        
        # Write command after each segment
        gwl_lines.append("Write")
    
    return '\n'.join(gwl_lines)


def generate_gwl_files(endpoint_data: Dict, gwl_params: Dict, output_dir: Path) -> List[Path]:
    """
    Generate GWL files from endpoint JSON
    
    Args:
        endpoint_data: Endpoint JSON with layers
        gwl_params: GWL printing parameters
        output_dir: Directory to save GWL files
        
    Returns:
        List of generated GWL file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gwl_files = []
    
    # Generate one GWL file per layer
    for layer_idx, layer in enumerate(endpoint_data['layers']):
        z = layer['z_um']
        
        # Generate GWL content
        gwl_content = serialize_layer_to_gwl(layer, gwl_params)
        
        # Create filename: layer_000_z0.00.gwl
        filename = f"layer_{layer_idx:03d}_z{z:.2f}.gwl"
        filepath = output_dir / filename
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(gwl_content)
        
        gwl_files.append(filepath)
    
    return gwl_files


def load_gwl_parameters(param_file: Path) -> Dict:
    """Load GWL-specific parameters from file"""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to float if possible
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    
    return params


def generate_master_gwl(gwl_files: List[Path], gwl_params: Dict, output_path: Path) -> Path:
    """
    Generate master GWL file that includes all layer files
    
    Args:
        gwl_files: List of layer GWL file paths
        gwl_params: GWL printing parameters
        output_path: Path for master GWL file
        
    Returns:
        Path to generated master GWL file
    """
    lines = []
    
    # Add writing parameters - read from config
    power_scaling = gwl_params.get('power_scaling', 1.0)
    lines.append(f"PowerScaling {power_scaling}")
    lines.append("")
    
    # Add variable definitions
    laser_power = int(gwl_params.get('laser_power', 40))
    scan_speed = int(gwl_params.get('scan_speed', 100000))
    interface_pos = gwl_params.get('find_interface_at', 0.5)
    
    lines.append(f"var $solidLaserPower = {laser_power}")
    lines.append(f"var $solidScanSpeed = {scan_speed}")
    lines.append("")
    lines.append("var $baseLaserPower = $solidLaserPower")
    lines.append("var $baseScanSpeed = $solidScanSpeed")
    lines.append("")
    lines.append(f"var $interfacePos = {interface_pos}")
    lines.append("")
    
    # Include each layer file (from bottom to top)
    for gwl_file in gwl_files:
        lines.append(f"include {gwl_file.name}")
    
    # Write master file
    master_content = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(master_content)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python gwl_serializer.py <endpoints.json> <output_dir>")
        print("Example: python gwl_serializer.py endpoints.json gwl_output/")
        sys.exit(1)
    
    # Load endpoint JSON
    endpoints_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    with open(endpoints_file, 'r') as f:
        endpoint_data = json.load(f)
    
    # Load GWL parameters
    script_dir = Path(__file__).parent
    param_file = script_dir / "PrintParameters.txt"
    gwl_params = load_gwl_parameters(param_file)
    
    # Generate GWL files
    print(f"Generating GWL files from {endpoints_file}...")
    gwl_files = generate_gwl_files(endpoint_data, gwl_params, output_dir)
    
    print(f"[OK] Generated {len(gwl_files)} GWL files in {output_dir}/")
    print(f"  Z range: {endpoint_data['layers'][0]['z_um']:.2f} to {endpoint_data['layers'][-1]['z_um']:.2f} um")
    print(f"  Files: {gwl_files[0].name} ... {gwl_files[-1].name}")
