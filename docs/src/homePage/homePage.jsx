import styles from "./homePage.module.css"

function Links(links=[{title:"title",description:"description"},{title:"title",description:"description"}]){
    const leftAmount = Math.ceil(links.length/2);
    const rightAmount =links.length-leftAmount;
    const left = links.slice(0,leftAmount);
    const right = links.slice(rightAmount,links.length);
    const listRight = right.map(link=><li><p>{link.title}</p><p>{link.description}</p></li>);
    const listLeft = left.map(link=><li><p>{link.title}</p><p>{link.description}</p></li>);
    return(
        <div>
            <div>
                <ul>
                    {listRight}
                </ul>
            </div>
            <div>
                <ul>
                    {listLeft}
                </ul>
            </div>
        </div>
    );
}

function Contentbody(){
    return(
        <div className={styles.Contentbody} id="home">
            <h1>Enhaning Explainability in Large Language Models documentation</h1>
            <p>Welcome! this is the documentation EELLM 1.0 </p>
            <h2>Documentation Sections:</h2>
            <Links/>
        </div>
    );
}

export default Contentbody