import clover from "../assets/clover.ico"
import styles from "./Topbar.module.css"
function Dropdown({name="NAME",id="Dropdown", isSelected=false, defaultSelection="two", options=["one","two","three"]}){
    const listOptions = options.map(option => <option key={option} value={option} >{option}</option>)
    return(
        <select name={name} id={id} defaultValue={isSelected?defaultSelection:options[0]}>
            {listOptions}
        </select>
    );
}

function Topbar(){
    return(
        <div className={styles.Topbar}>
            <div id={styles.left}>
                <a href="https://github.com/danez13/Enhancing-Explainablity-In-LLM"><img src={clover} alt="four lead clover" /></a>
                <p><a href="https://github.com/danez13/Enhancing-Explainablity-In-LLM">EELLM</a></p>
                <Dropdown 
                    name="Languages"
                    isSelected={true}
                    defaultSelection="English"
                    options={["English","Spanish"]}/>
                <Dropdown 
                    name="Versions"
                    isSelected={true}
                    defaultSelection="1.0"
                    options={["1.0"]}/>
            </div>
            <div id={styles.right}>
                <p>Theme</p>
                <Dropdown
                    name="Theme"
                    isSelected={true}
                    defaultSelection="Dark"
                    options={["Light","Dark"]}/>
                <p>|</p>
                <p><a href="">Modules</a></p>
                <p>|</p>
                <p><a href="">Index</a></p>
            </div>
        </div>
    );
}

export default Topbar